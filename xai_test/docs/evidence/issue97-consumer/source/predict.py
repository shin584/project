"""
predict.py
──────────
모델 로딩 및 절단 효율 예측 전담 모듈.
"""

import numpy as np
import os
import tensorflow as tf
import torch
import zipfile  # 💡 Keras 3 경로 버그와 TypeError를 동시에 해결하기 위해 추가
import streamlit as st
from pam_scanner import gaussian_penalty, ScanResult, VariantResult
from adapters import OneHotAdapter
from backbones import CNN_RNN_Backbone, IntegratedPredictor

# SpCas9 변종 이름 (모델 출력 인덱스 순서와 일치)
SPCAS9_VARIANTS = [
    'SpCas9(WT)', 'SpCas9-NG', 'VRQR', 'xCas9',
    'Sniper', 'SpCas9-HF1', 'eSpCas9(1.1)', 'HypaCas9', 'evoCas9'
]

MODEL_SCORE_SCALE = {
    'SpCas9': 100,
    'SaCas9': 100,
    'Cas12a': 1,
}

@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(BASE_DIR, "models", "Multi-Cas_1IN_9Hydra_Divide_Testset.keras")
    cas12a_path = os.path.join(BASE_DIR, "models", "cas12a_model_fold_4.keras")
    
    model = tf.keras.models.load_model(model_path, compile=False)
    model_cas12a = tf.keras.models.load_model(cas12a_path, compile=False)

    # 파이토치 모델 로드 (기존 동일)
    weight_files = [
        "models/crispr-cleavage-predictor/best_model_fold1.pth",
        "models/crispr-cleavage-predictor/best_model_fold2.pth",
        "models/crispr-cleavage-predictor/best_model_fold3.pth",
        "models/crispr-cleavage-predictor/best_model_fold4.pth",
    ]

    models_sa = []
    for w_path in weight_files:
        full_w_path = os.path.join(BASE_DIR, w_path)
        adapter  = OneHotAdapter(input_dim=4, hidden_dim=128)
        backbone = CNN_RNN_Backbone(input_dim=128, lstm_hidden=128, dropout=0.0)
        m        = IntegratedPredictor(adapter=adapter, backbone=backbone)
        m.load_state_dict(torch.load(full_w_path, map_location="cpu"))
        m.eval()
        models_sa.append(m)

    return model, models_sa, model_cas12a


def one_hot_encode_dna(sequence: str) -> np.ndarray:
    mapping = {
        'A': [1,0,0,0], 'C': [0,1,0,0],
        'G': [0,0,1,0], 'T': [0,0,0,1],
    }
    return np.array([mapping.get(b, [0,0,0,0]) for b in sequence.upper()])


def extract_seq_features(seq: str) -> tuple[np.ndarray, np.ndarray]:
    protospacer = seq[8:31]
    gc_total = (protospacer.count('G') + protospacer.count('C')) / 23
    gc_seed  = (protospacer[-8:].count('G') + protospacer[-8:].count('C')) / 8
    dinuc_map = {
        a + b: i
        for i, (a, b) in enumerate([(x, y) for x in 'ACGT' for y in 'ACGT'])
    }
    dinuc_features = np.zeros(22 * 16, dtype=np.float32)
    for i in range(22):
        pair = protospacer[i:i + 2]
        if pair in dinuc_map:
            dinuc_features[i * 16 + dinuc_map[pair]] = 1.0
    return np.array([gc_total, gc_seed], dtype=np.float32), dinuc_features


def build_cas12a_inputs(sequence: str) -> tuple[np.ndarray, np.ndarray]:
    seq_input = np.expand_dims(one_hot_encode_dna(sequence).astype(np.float32), axis=0)
    gc_feat, dinuc_feat = extract_seq_features(sequence)
    feat_input = np.expand_dims(np.concatenate([gc_feat, dinuc_feat]), axis=0)
    return seq_input, feat_input


def run_prediction(all_results: ScanResult, model, models_sa, model_cas12a) -> ScanResult:
    """
    모든 CandidateSite에 대해 모델 예측을 수행.
    """
    all_results.variant_results = []  # 초기화

    for site in all_results.sites:
        if not site.model_input_seq:
            continue

        X_np    = np.expand_dims(one_hot_encode_dna(site.model_input_seq), axis=0)
        penalty = gaussian_penalty(site.distance)

        # ── SpCas9: 9개 변종 개별 처리 ──────────────────────────────
        if site.cas_type == 'SpCas9':
            preds = model.predict(X_np, verbose=0)

            best_score = -1.0
            for i, variant_name in enumerate(SPCAS9_VARIANTS):
                raw         = float(preds[i][0][0])
                final_score = raw * penalty * MODEL_SCORE_SCALE['SpCas9']

                all_results.variant_results.append(VariantResult(
                    cas_name    = variant_name,
                    pam_seq     = site.pam_seq,
                    guide_seq   = site.guide_seq,
                    distance    = site.distance,
                    strand      = site.strand,
                    raw_score   = raw,
                    final_score = final_score,
                ))

                if final_score > best_score:
                    best_score   = final_score
                    best_raw     = raw

            site.raw_score   = best_raw
            site.final_score = best_score

        # ── SaCas9: PyTorch 5-Fold 앙상블 ───────────────────────────
        elif site.cas_type == 'SaCas9':
            X_tensor = torch.tensor(X_np, dtype=torch.float32)
            scores   = []
            with torch.no_grad():
                for m in models_sa:
                    scores.append(float(m(X_tensor).item()))
            raw         = sum(scores) / len(scores)
            final_score = raw * penalty * MODEL_SCORE_SCALE['SaCas9']

            site.raw_score   = raw
            site.final_score = final_score

            all_results.variant_results.append(VariantResult(
                cas_name    = 'SaCas9',
                pam_seq     = site.pam_seq,
                guide_seq   = site.guide_seq,
                distance    = site.distance,
                strand      = site.strand,
                raw_score   = raw,
                final_score = final_score,
            ))

        # ── Cas12a ───────────────────────────────────────────────────
        elif site.cas_type == 'Cas12a':
            cas12a_seq_input, cas12a_feat_input = build_cas12a_inputs(site.model_input_seq)
            raw         = float(model_cas12a.predict([cas12a_seq_input, cas12a_feat_input], verbose=0)[0][0])
            final_score = raw * penalty * MODEL_SCORE_SCALE['Cas12a']

            site.raw_score   = raw
            site.final_score = final_score

            all_results.variant_results.append(VariantResult(
                cas_name    = 'Cas12a',
                pam_seq     = site.pam_seq,
                guide_seq   = site.guide_seq,
                distance    = site.distance,
                strand      = site.strand,
                raw_score   = raw,
                final_score = final_score,
            ))

    return all_results