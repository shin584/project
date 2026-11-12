"""시연 서열의 SHAP을 미리 계산해 data/precomputed_shap/에 저장한다. (#152)

가중치가 있어야 돌아간다. 한 번 돌려 두면 대시보드에서 같은 서열 + 기본 설정
(샘플 64 / 배경 4)으로 SHAP을 열 때 계산 없이 즉시 결과가 나온다.

사용법
    python make_precomputed_shap.py 서열1 [서열2 ...]

서열은 대시보드에 넣는 것과 같은 81bp DNA다. 스캐너가 그 안에서 사이트를 찾고,
Cas12a 사이트의 모델 입력 윈도우에 대해 SHAP을 계산한다.
"""

import sys

import pam_scanner
import predict
from xai import explain, shap_cache

# 대시보드 입력칸 기본값. 다른 값으로 만들면 시연 경로에서 미적중이 난다.
NSAMPLES = 64
N_BACKGROUND = 4

# 사전계산할 Cas 종류. 워밍업과 같은 범위로 둔다.
TARGET_CAS = ("Cas12a",)


def main(seqs: list[str]) -> int:
    if not seqs:
        print(__doc__)
        return 1

    predictors = predict.load_predictors()
    made = skipped = failed = 0

    for n, raw in enumerate(seqs, 1):
        print(f"\n[{n}/{len(seqs)}] {raw[:30]}...")
        try:
            scan = pam_scanner.scan_all(raw)
        except ValueError as e:
            print(f"  건너뜀 — {e}")
            failed += 1
            continue

        sites = [s for s in scan.sites if s.cas_type in TARGET_CAS]
        if not sites:
            print(f"  {'/'.join(TARGET_CAS)} 사이트가 없습니다")
            skipped += 1
            continue

        for site in sites:
            predictor = predictors.get(("A", site.cas_type))
            if predictor is None or not predictor.connected:
                print(f"  {site.cas_type} 모델이 연동돼 있지 않습니다")
                failed += 1
                continue

            window = site.model_input_seq
            for head in predictor.explain_heads(site):
                key = shap_cache.shap_cache_key(
                    f"A/{site.cas_type}", head, window, NSAMPLES, N_BACKGROUND)
                if shap_cache.load(key) is not None:
                    print(f"  이미 있음 — {head} {window[:12]}...")
                    skipped += 1
                    continue

                print(f"  계산 중 — {head} {window[:12]}... (30초쯤 걸립니다)")
                result = explain.kernel_shap(
                    predictor.explain_handle(head), window,
                    nsamples=NSAMPLES, n_background=N_BACKGROUND)
                path = shap_cache.save(key, result)
                print(f"  저장 → {path.name}")
                made += 1

    print(f"\n새로 만듦 {made} · 건너뜀 {skipped} · 실패 {failed}")
    print(f"저장 위치: {shap_cache.DEFAULT_DIR}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main([a for a in sys.argv[1:] if a.strip()]))