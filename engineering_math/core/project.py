"""GUI와 독립적인 설정 파일 및 CSV 묶음 저장입니다."""
import csv
import io
import json
import zipfile
import math
from pathlib import Path
from .models import validate_params
from engineering_math.topics import get_topic


def settings(topic, params):
    return {'version': 2, 'topic': topic, 'params': validate_params(get_topic(topic).inputs, params)}


def load_settings(path):
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict):
        raise ValueError('설정 파일은 JSON 객체여야 합니다.')
    if data.get('version') == 1:
        params = {key: data[key] for key in ('function', 'L', 'N', 'num_points', 'comparison_N') if key in data}
        params['comparison_N'] = ','.join(map(str, params.get('comparison_N', [])))
        params.update(data.get('advanced', {}))
        for key in ('center', 'width', 'frequency', 'rate'):
            if key in params:
                params[key] = float(params[key])
        if 'M' in params:
            params['M'] = int(params['M'])
        data = {'version': 2, 'topic': 'fourier', 'params': params}
    if data.get('version') != 2:
        raise ValueError('지원하지 않는 설정 버전입니다.')
    return settings(data['topic'], data['params'])


def save_settings(path, topic, params):
    Path(path).write_text(json.dumps(settings(topic, params), ensure_ascii=False, indent=2), encoding='utf-8')


def export_results(path, result):
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('settings.json', json.dumps(settings(result.topic, result.params), ensure_ascii=False, indent=2))
        archive.writestr('metrics.json', json.dumps(result.metrics, ensure_ascii=False, indent=2))
        for name, (columns, rows) in result.tables.items():
            # 전체 CSV 문자열을 메모리에 복제하지 않고 ZIP 항목에 순서대로 씁니다.
            with archive.open(name+'.csv', 'w') as raw:
                with io.TextIOWrapper(raw, encoding='utf-8-sig', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(columns)
                    writer.writerows(['' if isinstance(value, float) and not math.isfinite(value) else value for value in row] for row in rows)
