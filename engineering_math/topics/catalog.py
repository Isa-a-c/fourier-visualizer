"""수학 라이브러리와 독립적인 문제·영역·해법 지원 명세입니다."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Experiment:
    topic: str
    module: str
    class_name: str
    lesson: str
    group: str
    problem: str
    configuration: str
    method: str
    defaults: dict = field(default_factory=dict)


GROUPS = ('함수 전개·근사', '고유값·고유함수', '적분변환·표본화', '진동·ODE', 'PDE·공간 분포', '수식·해 검산')
SERIES, EIGEN, TRANSFORM, ODE, PDE, VERIFY = GROUPS
EXPERIMENTS = (
    Experiment('fourier', 'fourier', 'FourierTopic', '11.1 · Fourier 급수', SERIES, '함수 전개', '전체 구간 · 삼각함수', 'Fourier 직교투영'),
    Experiment('half_range', 'chapter11_series', 'HalfRangeTopic', '11.2 · 반구간 전개', SERIES, '함수 전개', '반구간 · 사인/코사인', '홀수·짝수 확장 후 투영'),
    Experiment('fourier_forced', 'chapter11_series', 'ForcedTopic', '11.3 · Fourier 강제진동', ODE, '질량–스프링–댐퍼', '주기 외력 · Fourier 근사', '유한 급수 외력의 선형 응답'),
    Experiment('trig_approximation', 'chapter11_series', 'ApproximationTopic', '11.4 · 최소제곱 근사', SERIES, '함수 전개', '삼각다항식 · 계수 변형 검산', '직교투영과 적분 제곱오차 비교'),
    Experiment('sturm_liouville', 'chapter11_series', 'SturmTopic', '11.5 · Sturm–Liouville', EIGEN, '상수계수 경계값 문제', '유한 구간 · DD/NN/DN', '고유함수와 가중 투영'),
    Experiment('orthogonal_series', 'chapter11_series', 'OrthogonalTopic', '11.6 · 직교급수', SERIES, '함수 전개', 'Legendre / Bessel 기저', '가중 직교투영'),
    Experiment('fourier_integral', 'chapter11_transforms', 'IntegralTopic', '11.7 · Fourier 적분', TRANSFORM, 'Fourier 변환', '전체 실수축 · A/B 표현', '유한 구간 수치 구적'),
    Experiment('sine_cosine_transform', 'chapter11_transforms', 'SineCosineTopic', '11.8 · 사인·코사인 변환', TRANSFORM, 'Fourier 변환', '양의 반축 · 사인/코사인', '유한 구간 수치 구적'),
    Experiment('continuous_fourier', 'chapter11_transforms', 'ContinuousTransformTopic', '11.9 · 연속 변환·FFT', TRANSFORM, 'Fourier 변환', '복소 스펙트럼 · FFT 비교', '수치 구적과 FFT 비교'),
    Experiment('transform_table', 'chapter11_transforms', 'TransformTableTopic', '11.10 · 변환공식 검산', TRANSFORM, 'Fourier 변환', '대표 공식 · 수치 검산', '공식과 수치 구적 비교'),
    Experiment('pde_basics', 'chapter12', 'PdeBasicsTopic', '12.1 · PDE 해 검산', VERIFY, 'PDE 후보 해', '기호 미분 · 표본 잔차', '후보 함수의 방정식 잔차'),
    Experiment('wave', 'wave', 'WaveTopic', '12.2–12.3 · 고정단 현', PDE, '파동', '유한 현 · 양 끝 고정', '사인 모드의 시간 진화'),
    Experiment('dalembert', 'chapter12', 'DalembertTopic', '12.4 · D’Alembert', PDE, '파동', '무한 현 · 초기값', 'D’Alembert 공식과 수치 구적'),
    Experiment('heat', 'heat', 'HeatTopic', '12.5–12.6 · 유한 막대', PDE, '열전도', '유한 막대 · 고정 온도', '정상 상태 분리와 사인 급수'),
    Experiment('infinite_heat', 'chapter12', 'InfiniteHeatTopic', '12.7 · 무한 막대', PDE, '열전도', '무한 막대 · 초기값', '열핵과 Gauss–Hermite 구적'),
    Experiment('rectangle_membrane', 'chapter12', 'MembraneTopic', '12.8–12.9 · 직사각형 막', PDE, '파동', '직사각형 막 · 가장자리 고정', '이중 사인 모드'),
    Experiment('disk_membrane', 'chapter12', 'DiskMembraneTopic', '12.10 · 원형 막', PDE, '파동', '원형 막 · 원주 고정', 'Fourier–Bessel 모드'),
    Experiment('coordinate_potential', 'chapter12', 'PotentialTopic', '12.11 · 퍼텐셜', PDE, '정상 상태 퍼텐셜', '원판 / 축대칭 구 · Dirichlet', '조화함수의 경계 전개'),
    Experiment('pde_laplace', 'chapter12', 'PdeLaplaceTopic', '12.12 · Laplace PDE', PDE, '파동', '반무한 현 · 초기값 0', 'Laplace 시간 지연 공식', {'mode': '반무한 현'}),
    Experiment('ode', 'ode', 'OdeTopic', 'ODE · 질량–스프링–댐퍼', ODE, '질량–스프링–댐퍼', '직접 외력 · 초기값', '수치 적분 및 지원 외력의 독립 검산'),
    Experiment('laplace', 'laplace', 'LaplaceTopic', 'Laplace 변환·역변환', TRANSFORM, 'Laplace 변환', '단측 변환 · 기호 계산', 'SymPy 변환과 수렴 조건'),
    Experiment('pde_laplace', 'chapter12', 'PdeLaplaceTopic', '12.12 · 반무한 열전도', PDE, '열전도', '반무한 막대 · 계단 경계·초기값 0', 'Laplace 기반 erfc 해', {'mode': '반무한 열전도'}),
)

NUMERICAL_KEYS = frozenset({'num_points', 'time_points', 'frequency_points', 'quadrature_points',
                          'radial_points', 'angular_points', 'quadrature_order', 'order',
                          'max_step', 'breakpoints', 'comparison_alpha'})


def experiment_index(topic_id, params=None):
    matches = [i for i, item in enumerate(EXPERIMENTS) if item.topic == topic_id]
    if not matches:
        raise ValueError('등록되지 않은 주제입니다.')
    if params:
        for i in matches:
            if all(params.get(key) == value for key, value in EXPERIMENTS[i].defaults.items()):
                return i
    return matches[0]
