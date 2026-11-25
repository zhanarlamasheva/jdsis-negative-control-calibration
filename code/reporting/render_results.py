"""Create a Russian Markdown report and an SVG from the completed CSV results."""
from pathlib import Path
import csv
import json
import html
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'


def main():
    rows=list(csv.DictReader((OUT/'summary.csv').open(encoding='utf-8')))
    def get(s, m='sign_mom', theta=0.):
        return next(r for r in rows if r['scenario']==s and r['method']==m and float(r['true_effect'])==theta)
    def f(row, key): return float(row[key])
    # Paired checks on complete summaries, plus population-vs-Monte-Carlo oracle.
    for theta in [0., .1]:
        baseline=get('all_units_1','standardized_sign_mom',theta)
        for scenario in ['all_units_0.1','all_units_10','one_control_units_10','affine_shift_100','reverse_all_units']:
            row=get(scenario,'standardized_sign_mom',theta)
            for key in ['bias','rmse','coverage','mean_width','rejection_rate']:
                assert np.isclose(f(row,key),f(baseline,key),atol=1e-10), (scenario,key)
    theory=list(csv.DictReader((OUT/'population_truth.csv').open(encoding='utf-8')))
    for t in theory:
        r=get(t['scenario'],'oracle_sign_mom',float(t['true_effect']))
        assert abs(f(r,'bias')-f(t,'oracle_sign_residual'))<5*f(r,'bias_mcse')
    checks={'standardization_affine_invariance':'passed across all paired summaries',
            'oracle_sign_population_bias':'all scenarios within 5 Monte Carlo SE',
            'summary_rows':len(rows)}
    (OUT/'analysis_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    groups=[('1. Величины смещения', [('magnitude_ratio_0.25','Контроли ×0.25'),('magnitude_ratio_1','Равные величины'),('magnitude_ratio_2','Контроли ×2'),('magnitude_ratio_4','Контроли ×4'),('heterogeneous_mean_matched','Разные, среднее совпало')]),
            ('2. Два скрытых фактора', [('two_factors_aligned','Прокси согласован'),('two_factors_proxy_blind','Прокси задаёт обратный знак'),('two_factors_proxy_mixed','Прокси ошибается у половины')]),
            ('3. Единицы измерения', [('all_units_0.1','Все контроли ×0.1'),('all_units_1','Исходная шкала'),('all_units_10','Все контроли ×10'),('one_control_units_10','Один контроль ×10')])]
    methods=[('uncalibrated','Без коррекции','#888888'),('sign_mom','Ориентация знаков','#c43b3b'),('standardized_sign_mom','Стандартизация + знаки','#237db1')]
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="1180" viewBox="0 0 1280 1180">',
         '<rect width="1280" height="1180" fill="white"/>',
         '<g font-family="Arial, sans-serif" fill="#222">',
         '<text x="40" y="35" font-size="24">Стресс-тесты калибровки: смещение при истинном эффекте 0</text>',
         '<text x="40" y="62" font-size="15">1000 повторений, n=4000; непрерывный искусственный исход; отрезки: ±1.96 MCSE</text>']
    for j,(_,label,color) in enumerate(methods):
        x=40+j*390
        svg.extend([f'<rect x="{x}" y="82" width="15" height="15" fill="{color}"/>',f'<text x="{x+24}" y="95" font-size="15">{label}</text>'])
    x0=340; x1=1210; low=-1.1; high=.3
    def px(v): return x0+(v-low)/(high-low)*(x1-x0)
    top=135
    for title, scenarios in groups:
        svg.append(f'<text x="40" y="{top}" font-size="20" font-weight="bold">{title}</text>')
        start=top+20; end=start+len(scenarios)*65
        for tick in [-1.,-.8,-.6,-.4,-.2,0,.2]:
            x=px(tick)
            svg.append(f'<line x1="{x}" y1="{start}" x2="{x}" y2="{end}" stroke="{"#555" if tick==0 else "#ddd"}"/>')
            svg.append(f'<text x="{x}" y="{end+20}" text-anchor="middle" font-size="13">{tick:g}</text>')
        for i,(scenario,label) in enumerate(scenarios):
            yy=start+20+i*65
            svg.append(f'<text x="40" y="{yy+6}" font-size="15">{html.escape(label)}</text>')
            for j,(method,_,color) in enumerate(methods):
                row=get(scenario,method); bias=f(row,'bias'); err=1.96*f(row,'bias_mcse'); y=yy-10+j*13
                svg.append(f'<line x1="{px(bias-err)}" y1="{y}" x2="{px(bias+err)}" y2="{y}" stroke="{color}" stroke-width="2"/>')
                svg.append(f'<circle cx="{px(bias)}" cy="{y}" r="4" fill="{color}"/>')
        top=end+65
    svg.extend(['<text x="40" y="1160" font-size="14">Стандартизация устраняет зависимость от единиц, но не доказывает сопоставимость смещений.</text>','</g></svg>'])
    (OUT/'bias_stress_tests.svg').write_text('\n'.join(svg),encoding='utf-8')
    lines=['# Результаты проверки устойчивости', '',
           '29 октября 2026 г. Проверка выполнена на искусственных данных, без OULAD/XES. Рукопись не изменялась.', '',
           '## Главный результат', '',
           'Одной ориентации знаков недостаточно для надёжного переноса смещения с контролей на основной исход. Она работает в благоприятной модели с равными модулями смещений, но нарушается при несовпадении средних величин, неполном прокси нескольких конфаундеров и смене единиц измерения контролей. Это поддерживает необходимость переработать центральное методологическое утверждение, а не просто добавить подтверждающую симуляцию.', '',
           '![Смещение оценок](results/bias_stress_tests.svg)', '',
           '## Устройство проверки', '',
           '16 сценариев × 2 истинных эффекта (0 и 0.1) × 1000 повторений; в каждом повторении 4000 наблюдений и 6 отрицательных контролей. Для каждого набора данных сравниваются 6 процедур: всего 192000 записей оценок, 192 сводные строки. Это 32000 наборов данных с общими случайными числами между сценариями, а не 32000 независимых исследований. Генеративная модель и все параметры описаны в README.md и results/config.json.', '',
           'Оцениватель — разность средних. Проверяется арифметический шаг калибровки из рукописи. Непрерывный Y позволяет точно вычислить истинный эффект и остаточное смещение. Результаты не являются процентными пунктами retention и не заменяют AIPW-проверку на бинарных исходах.', '',
           '## Основные числа: истинный эффект 0', '',
           'Покрытие — доля интервалов, включивших истинный эффект; ошибка I рода — доля интервалов, исключивших ноль. «±» у смещения означает одну MCSE, а не стандартное отклонение отдельных оценок.', '',
           '| Сценарий | Процедура | Смещение ± MCSE | RMSE | Покрытие 95% | Ошибка I рода | Ширина |',
           '|---|---|---:|---:|---:|---:|---:|']
    selection=['magnitude_ratio_1','magnitude_ratio_2','magnitude_ratio_4','heterogeneous_mean_matched',
               'two_factors_proxy_blind','two_factors_proxy_mixed','all_units_0.1','all_units_10','one_control_units_10']
    labels={s:l for _,ss in groups for s,l in ss}
    for scenario in selection:
        for method,label in [('uncalibrated','Без коррекции'),('sign_mom','Знаки'),('standardized_sign_mom','Стандартизация + знаки')]:
            r=get(scenario,method)
            lines.append(f"| {labels.get(scenario,scenario)} | {label} | {f(r,'bias'):+.4f} ± {f(r,'bias_mcse'):.4f} | {f(r,'rmse'):.4f} | {f(r,'coverage'):.3f} | {f(r,'rejection_rate'):.3f} | {f(r,'mean_width'):.3f} |")
    lines+=['', '## Восстановление ненулевого эффекта 0.1', '',
            '| Сценарий | Процедура | Средняя оценка | Смещение | Покрытие | Мощность |',
            '|---|---|---:|---:|---:|---:|']
    for scenario in ['magnitude_ratio_1','magnitude_ratio_2','two_factors_proxy_blind','all_units_10']:
        for method,label in [('sign_mom','Знаки'),('standardized_sign_mom','Стандартизация + знаки'),('oracle_transfer_covariance','Известный перенос (oracle)')]:
            r=get(scenario,method,.1)
            lines.append(f"| {labels[scenario]} | {label} | {.1+f(r,'bias'):+.4f} | {f(r,'bias'):+.4f} | {f(r,'coverage'):.3f} | {f(r,'rejection_rate'):.3f} |")
    lines+=['', '## Интерпретация трёх экспериментов', '',
            '1. **Величина смещения.** При bY>0 и правильных знаках остаток равен bY−mean(|bk|). Нагрузки могут различаться без смещения, если средняя величина совпадает с целевой; равенство средних — дополнительное условие, которое ориентация не устанавливает. Раздел 3.5 рукописи ошибочно распространяет точное устранение смещения на произвольные bk. Удвоение средней силы контроля оставляет остаток −bY, даже при известном правильном знаке.', '',
            '2. **Несколько факторов.** В неблагоприятном сценарии модули смещений специально оставлены равными целевому. Контроль имеет нагрузку (−0.1,0.2), а лечение зависит от U1+U2. Прокси U1+noise коррелирует с контролем отрицательно, хотя суммарное смещение положительно. Поэтому сбой вызван именно ориентацией по неполному прокси. Это конструктивный контрпример, а не оценка частоты такого события в образовательных данных.', '',
            '3. **Единицы.** Умножение всех контролей на 10 изменяет предельную коррекцию в 10 раз, хотя причинное устройство и информация не изменились. Сдвиг на 100 и обращение знаков, напротив, сохраняют результат sign-aware, как и должны. Стандартизация устраняет зависимость от аффинных шкал, но не гарантирует равенства конфаундинга: дисперсия контроля включает специфический шум, а не только общий скрытый фактор.', '',
            '## Проверки и неопределённость', '',
            '- Пройдены аналитические проверки равных/неравных нагрузок, смены единиц, обращения знаков и аффинной инвариантности стандартизации.',
            '- Смещения oracle-sign во всех сценариях находятся в пределах 5 MCSE от точных популяционных значений; oracle-transfer — в пределах 5 MCSE от нулевого смещения.',
            '- При истинном эффекте 0 ошибка I рода равна 1−покрытие, поскольку оба события получены из одного двухстороннего интервала.',
            '- При 1000 повторениях MCSE доли около 0.95 равна примерно 0.0069. Малые отклонения от 0.95 не следует объявлять доказательством нарушения покрытия.',
            '- MoM-интервалы — plug-in-формула рукописи: неопределённость ориентации, стандартизации и параметров эмпирического нуля отдельно не распространяется. Поведение этих интервалов не характеризует все существующие методы эмпирической калибровки.', '',
            '## Что это означает для переработки статьи', '',
            'Текущая проверка не устанавливает новый метод. Она даёт основание заменить безусловное утверждение о восстановлении смещения на условия переносимости и контрпримеры. Термин «исправление канонической калибровки» требует отдельной проверки по оригинальным публикациям и реализации пакета.', '',
            'Перспективный следующий этап — обосновать коэффициенты переноса смещения или диапазоны чувствительности к ним. Нельзя оценить bY/bk только по знакам корреляций. Oracle-transfer в таблицах использует истинные параметры генеративной модели; он демонстрирует, какая информация нужна, но не является практически доступным решением.', '',
            'Для полноценного методологического результата далее нужны: идентификационные предпосылки; корректное распространение неопределённости; бинарные/нелинейные DGP и AIPW; реальные контроли в полусинтетических данных; сопоставление с существующими методами переноса и калибровки. Следует оценивать честные интервалы и устойчивость, а не добиваться желаемой значимости.', '',
            '## Воспроизводимость', '',
            'Запуск и протокол: README.md. Полные числа: results/summary.csv. Отдельные повторения: results/replications.csv. Аналитическая истина: results/population_truth.csv. Параметры, версии и SHA-256 исходного расчётного скрипта: results/config.json. Хеши таблиц и длительность: results/manifest.json. Дополнительные проверки: results/analysis_checks.json.', '']
    (ROOT/'REPORT_RU.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(checks,indent=2))
    for scenario in ['magnitude_ratio_1','magnitude_ratio_2','two_factors_proxy_blind','all_units_10']:
        r=get(scenario)
        print(scenario, {k:r[k] for k in ['bias','coverage','rejection_rate','mean_width','sign_error_rate']})


if __name__=='__main__':
    main()
