# 외부 sampler의 실제 프로세스 점검

실제 명령은 다음과 같고 `launch.json`에 원래 argv·cwd를 보존했다.

```powershell
py -3.13 -X utf8 docs/benchmarks/measure_ram.py watch .scratch/issue203/sampler-smoke01 -- python .scratch/issue203/sample_workload.py
```

당시의 `sample_workload.py`와 raw samples·sampler 결과·로그를 이 폴더에 복사했다.
새 빈 출력 경로와 이 폴더의 workload 경로로 다시 실행할 수 있다. 부모 8MiB와 자식
24MiB를 실제 할당하고 정상 종료한다. 검사는 **3 통과 / 0 실패 / 0 건너뜀**이다.
최초 질의의 총 프로세스 수 2 가정은 conhost 때문에 실패했다(2 통과 / 1 실패).
sampler가 올바르게 추적한 conhost를 제외하고 Python 두 개를 확인하도록 질의를 고쳤다.

실제 검증 명령의 내용은 아래와 같다. `p`는 원래 `.scratch/issue203/sampler-smoke01`였고
보관 자료를 확인할 때는 이 폴더로 바꾸면 된다.

```python
import gzip
import json
from pathlib import Path

p = Path("docs/benchmarks/issue203-sampler-smoke")
meta = json.loads((p / "sampler.json").read_text())
rows = [json.loads(line) for line in gzip.open(p / "samples.jsonl.gz", "rt")]
checks = {
    "two_python_processes": sum(x["name"] == "python.exe" for x in meta["process_inventory"].values()) == 2,
    "allocations_in_private_peak": max(x["private"] for x in rows) > 32 * 1024**2,
    "clean_exit": meta["exit_code"] == 0 and not meta["survivors"] and not meta["access_errors"],
}
assert all(checks.values()), checks
```
