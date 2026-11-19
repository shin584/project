"""#201 validation entry point; each test/model stage gets a fresh Python process.

The plan in .scratch/issue201/validation-plan.json selects only these fixed stages.
Keeping one entry point permits repeating authorized checks after repairs.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
TESTS = [
    'test/test_nt_explain.py', 'test/test_predict.py', 'test/test_explain.py',
    'test/test_explain_cache.py', 'test/test_cache_identity.py', 'test/test_shap_rng.py',
    'test/test_core_facade.py', 'tests/test_desktop_xai.py', 'tests/test_nt_cancellation.py',
    'tests/test_nt_xai_view.py',
    'tests/test_progress_callback.py', 'tests/test_shap_cache.py', 'tests/test_worker_cache.py',
]
STAGES = {
    'focused': ['-m', 'pytest', 'test/test_nt_explain.py', 'tests/test_nt_cancellation.py', '-q', '-p', 'no:cacheprovider'],
    'regression': ['-m', 'pytest', *TESTS, '-q', '-p', 'no:cacheprovider'],
    'SpCas9': ['docs/nt-xai/verify.py', 'SpCas9'],
    'Cas12a': ['docs/nt-xai/verify.py', 'Cas12a'],
    'process': ['-m', 'pytest', 'tests/test_nt_process.py', '-q', '-p', 'no:cacheprovider'],
    'audit': ['docs/nt-xai/audit_evidence.py'],
    'display': ['docs/nt-xai/render_evidence.py'],
    'shared_ig': ['-m', 'pytest',
        'test/test_cas12a_ig.py::test_selected_handle_explains_the_composed_model_through_the_existing_loop',
        'test/test_cas12a_ig.py::test_path_protocol_prevents_cross_path_memory_and_disk_cache_hits',
        'test/test_cas12a_ig.py::test_selected_path_keeps_progress_and_callback_cancellation',
        '-q', '-p', 'no:cacheprovider'],
}


if __name__ == '__main__':
    plan_path = ROOT / '.scratch/issue201/validation-plan.json'
    plan = (json.loads(plan_path.read_text(encoding='utf-8-sig')) if plan_path.exists()
            else ['regression', 'SpCas9', 'Cas12a', 'process'])
    result_path = ROOT / '.packaging/evidence/issue201/validation-results.json'
    result_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for stage in plan:
        command = [sys.executable, '-X', 'utf8', *STAGES[stage]]
        print('RUN', stage, ' '.join(command), flush=True)
        outcome = subprocess.run(command, check=False)
        results.append({'stage': stage, 'command': command, 'exit_code': outcome.returncode})
        result_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    raise SystemExit(int(any(r['exit_code'] for r in results)))
