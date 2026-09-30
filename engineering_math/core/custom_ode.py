"""사용자 외력의 구간 분할과 서로 다른 수치 적분 결과를 비교합니다."""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from .expr import parameter_expression, symbols_for, evaluate


def force_breakpoints(expression, parameter_text, manual_text, end_time):
    """선형 조건의 경계를 자동 발견하고 사용자가 지정한 경계를 합칩니다."""
    t = symbols_for(('t',))[0]
    points = {0., end_time}
    if manual_text.strip():
        for item in manual_text.split(';'):
            constant = parameter_expression(item, parameter_text, ('t',))
            if constant.free_symbols:
                raise ValueError('분할 시각은 t가 없는 실수 상수로 입력하십시오.')
            value = float(constant)
            if not np.isfinite(value) or not 0 <= value <= end_time:
                raise ValueError('분할 시각은 0과 마지막 시간 사이여야 합니다.')
            points.add(value)
    arguments = [node.args[0] for node in expression.atoms(sp.Heaviside, sp.sign, sp.Abs)]
    for piece in expression.atoms(sp.Piecewise):
        for _, condition in piece.args:
            for relation in sp.preorder_traversal(condition):
                if getattr(relation, 'is_Relational', False):
                    arguments.append(relation.lhs-relation.rhs)
    for argument in arguments:
        polynomial = argument.as_poly(t)
        if polynomial is not None and polynomial.degree() == 1:
            slope, offset = polynomial.all_coeffs()
            try:
                value = float(-offset/slope)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and 0 < value < end_time:
                points.add(value)
    if len(points) > 100:
        raise ValueError('시간 분할점은 100개 이하로 지정하십시오.')
    return sorted(points)


def solve_custom_force(params):
    """기호 변환에 의존하지 않고 수치해와 세분화 비교해를 먼저 구합니다."""
    expression = parameter_expression(params['force_expression'], params['parameters'], ('t',))
    if expression.has(sp.DiracDelta):
        raise ValueError('임펄스 외력은 일반 ODE 수치 적분으로 처리하지 않습니다. 라플라스 주제를 사용하십시오.')
    m, c, k = params['mass'], params['damping'], params['stiffness']
    time = np.linspace(0, params['end_time'], params['num_points'])
    boundaries = force_breakpoints(expression, params['parameters'], params['breakpoints'], params['end_time'])
    forces = evaluate(expression, (time,), ('t',))
    step = min(params['max_step'], .2/max(np.sqrt(k/m), c/m))
    if params['end_time']/(step/2) > 200000:
        raise ValueError('계산 규모가 큽니다. 마지막 시간을 줄이거나 최대 내부 간격을 늘리십시오.')

    def integrate(method, tolerance, max_step):
        output = np.empty((2, len(time)))
        initial = [params['position'], params['velocity']]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            def derivative(t, state):
                # 분할점에서는 해당 구간 안쪽의 극한 방향으로 외력을 평가합니다.
                inside = np.clip(t, np.nextafter(start, stop), np.nextafter(stop, start))
                force = float(evaluate(expression, (inside,), ('t',)))
                return [state[1], (force-c*state[1]-k*state[0])/m]
            solution = solve_ivp(derivative, (start, stop), initial, method=method,
                                 rtol=tolerance, atol=tolerance/100, max_step=max_step, dense_output=True)
            if not solution.success:
                raise ValueError('외력의 정의역·특이점 또는 적분 설정을 확인하십시오: '+solution.message)
            mask = (time >= start) & (time <= stop)
            if np.any(mask):
                output[:, mask] = solution.sol(time[mask])
            initial = solution.y[:, -1]
        if not np.all(np.isfinite(output)):
            raise ValueError('수치해에 NaN 또는 inf가 발생했습니다.')
        return output

    values = integrate('DOP853', 1e-8, step)
    reference = integrate('RK45', 1e-10, step/2)
    return expression, time, values, reference, forces, boundaries


def custom_ode_result(topic_id, params):
    from .models import Result
    expression, time, values, reference, forces, boundaries = solve_custom_force(params)
    m, c, k = params['mass'], params['damping'], params['stiffness']
    natural = np.sqrt(k/m)
    ratio = c/(2*np.sqrt(m*k))
    kind = '무감쇠' if c == 0 else '임계감쇠' if np.isclose(ratio, 1., rtol=1e-8, atol=1e-10) else '부족감쇠' if ratio < 1 else '과감쇠'
    energy = .5*m*values[1]**2+.5*k*values[0]**2
    if not np.all(np.isfinite(energy)):
        raise ValueError('에너지가 유한하지 않습니다. 외력의 크기를 줄이십시오.')
    notices = [
        f'감쇠 분류: {kind}. 외력 F(t) = {expression}',
        '수치 계산이 완료되었습니다. 비교해는 내부 간격과 허용 오차를 줄인 다른 적분법의 결과이며 해석해가 아닙니다.',
        '두 수치해가 일치해도 매우 좁은 펄스·빠른 진동·특이점을 놓칠 수 있습니다. 최대 내부 간격을 줄이고 분할 시각을 지정하십시오.',
        'Heaviside·Piecewise·sign·abs의 선형 조건 경계만 자동 분할합니다. 비선형·반복 불연속은 직접 분할 시각을 입력하십시오.',
    ]
    rows = [['F(t)', str(expression)], ['수치 계산', '완료 (DOP853 / 세분화 RK45 비교)'],
            ['분할 시각', '; '.join(map(str, boundaries))]]
    laplace_input = None
    if params['symbolic_mode'] == '라플라스 변환도 계산':
        t, s = symbols_for(('t',))[0], sp.Symbol('s')
        try:
            forcing, plane, condition = sp.laplace_transform(expression, t, s)
            mass, damping, stiffness, y0, v0 = map(lambda value: sp.Rational(str(value)),
                                                   (m, c, k, params['position'], params['velocity']))
            transformed = (forcing+mass*(s*y0+v0)+damping*y0)/(mass*s**2+damping*s+stiffness)
            rows.extend([['F(s)', str(forcing)], ['Y(s)', str(transformed)]])
            if forcing.has(sp.LaplaceTransform):
                rows.append(['기호 상태', '미평가 변환이 남았습니다. 수치 결과는 유지합니다.'])
                notices.append('라플라스 변환을 닫힌 형태로 구하지 못했습니다. 수치 결과와 미평가 식을 표시합니다.')
            else:
                rows.extend([['외력 변환 수렴 영역', f'Re(s) > {plane}'], ['추가 조건', str(condition)]])
                # 생성된 식도 입력 파서로 다시 읽을 수 있을 때만 연결 버튼을 제공합니다.
                try:
                    parameter_expression(str(transformed), '', ('s',))
                    laplace_input = str(transformed)
                    notices.append('라플라스 변환도 계산되었습니다. 연결 버튼으로 역변환을 요청할 수 있습니다.')
                except Exception:
                    notices.append('기호 변환식은 구했지만 현재 역변환 입력 문법으로 전달할 수 없어 연결 버튼을 생략합니다.')
        except Exception as exc:
            rows.append(['기호 상태', f'미완료 ({type(exc).__name__}); 수치 결과는 유지합니다.'])
            notices.append('기호 변환은 완료하지 못했지만 수치 결과는 유지합니다.')
    else:
        rows.append(['기호 상태', '미요청: 라플라스 변환도 계산을 선택하면 추가로 시도합니다.'])
        notices.append('라플라스 변환은 요청하지 않았습니다. 기호 계산 옵션에서 추가할 수 있습니다.')
    tables = {
        '시간 응답': (['t', 'y', 'velocity', 'refined y', 'force', 'energy'],
                    np.column_stack((time, values.T, reference[0], forces, energy))),
        '계산 가능 여부': (['항목', '결과'], rows),
    }
    return Result(topic_id, params,
                  {'고유 각주파수': natural, '감쇠비': ratio,
                   '세분화 변위 최대 차이': float(np.max(np.abs(values[0]-reference[0]))),
                   '세분화 속도 최대 차이': float(np.max(np.abs(values[1]-reference[1])))}, tables,
                  dict(time=time, values=values, reference=reference, energy=energy, force=forces,
                       reference_label='Refined RK45 (not analytic)', solution_label='DOP853',
                       laplace_input=laplace_input, expression=expression, boundaries=boundaries, regime=kind,
                       symbolic='\n\n'.join(f'{label}\n{value}' for label, value in rows)), notices)
