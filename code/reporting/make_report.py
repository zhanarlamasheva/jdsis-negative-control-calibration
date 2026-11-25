"""Report every predeclared scenario, including failures, from completed runs."""
from pathlib import Path
import csv
import hashlib
import json

ROOT=Path(__file__).resolve().parent


def read(folder):
    return list(csv.DictReader((ROOT/folder/'summary.csv').open(encoding='utf-8')))


def main():
    linear=read('results'); binary=read('binary_results')
    def num(row,key): return float(row[key])
    def find(rows,scenario,method,zero=True):
        return next(r for r in rows if r['scenario']==scenario and r['method']==method and
                    ((float(r.get('true_effect',r.get('alpha')))==0)==zero))
    for folder,script,protocol in [('results','run_validation.py','PROTOCOL_RU.md'),
                                   ('binary_results','run_binary_validation.py','BINARY_PROTOCOL_RU.md')]:
        config=json.loads((ROOT/folder/'config.json').read_text())
        assert config['source_sha256']==hashlib.sha256((ROOT/script).read_bytes()).hexdigest()
        assert config['protocol_sha256']==hashlib.sha256((ROOT/protocol).read_bytes()).hexdigest()
        manifest=json.loads((ROOT/folder/'manifest.json').read_text())
        for name,digest in manifest['sha256'].items():
            assert digest==hashlib.sha256((ROOT/folder/name).read_bytes()).hexdigest()
    # A fresh independent CSV-level audit of acceptance criteria and null identities.
    for row in linear+binary:
        assert 0<=num(row,'coverage')<=1
        if row.get('rejection_interpretation')=='type_I_error':
            assert abs(num(row,'coverage')+num(row,'rejection_rate')-1)<1e-12
        assert num(row,'rmse')+1e-12>=abs(num(row,'bias'))
    names=list(dict.fromkeys(r['scenario'] for r in linear))
    labels={'equal_loadings':'Равные нагрузки','half_loadings':'Контроли ×0.5',
            'double_loadings':'Контроли ×2','heterogeneous_loadings':'Разные нагрузки, среднее совпало',
            'changed_units':'Один контроль в другой шкале','noisy_controls':'Шумные контроли',
            'two_factors_aligned':'Два совместимых фактора','two_factors_incompatible':'Два несовместимых фактора',
            'reference_shift':'Нагрузки в эталоне изменены','weak_reference':'Слабый прокси и малый эталон',
            'invalid_controls':'Контроли имеют причинный эффект','no_confounding':'Нет скрытого конфаундинга',
            'small_sample':'n=500','large_sample':'n=8000',
            'matched':'Бинарные, одинаковые нагрузки','different_magnitudes':'Бинарные, разные нагрузки',
            'incompatible_factors':'Бинарные, несовместимые факторы'}
    methods=[('sign_mom','Исходные знаки + MoM'),('sign_covariance','Знаки + ковариация'),
             ('reference_transfer_fieller_hull','Перенос из эталона + Fieller')]
    def success(row): return row['conditional_success']=='True'
    success_counts={m:sum(success(r) for r in linear if r['method']==m) for m,_ in methods}
    lines=['# Проверка эффективности калибровки отрицательных контролей','',
           'Дата: 29 октября 2026 г. Все результаты получены на искусственных данных. Исходная рукопись и прежние стресс-тесты сохранены. Внешней пререгистрации нет; протоколы этой серии записаны до запуска соответствующих расчётов.','',
           '## Вывод','',
           'Эксперименты позволяют обосновать условную эффективность: ориентация знаков полезна, когда ориентированные смещения контролей сопоставимы с целевым. Ориентация сама по себе не устанавливает эту сопоставимость. Расширение с независимо оцениваемым коэффициентом переноса показывает, как можно восстановить эффект при разных величинах смещения, но использует дополнительную эталонную выборку и требует переносимости связей. Оно не является доказательством эффективности исходного метода без этих данных.','',
           'Выполнено 28000 линейных и 4000 бинарных симуляционных запусков, с общими случайными числами между сценариями. Сохранено 196000 линейных и 16000 бинарных записей оценок и интервалов. Набор параметров задан протоколами; неблагоприятные сценарии не исключались.','',
           '## Что считалось успехом','',
           'В конкретном сценарии требовались одновременно: положительная нижняя 95%-Monte-Carlo-граница снижения MSE относительно исходной оценки; покрытие 95%-интервалов в [0.93,0.97]; средняя ширина не более удвоенной исходной; для линейной серии — отсутствие неограниченных интервалов. Мощность показывается отдельно. Это операционное правило этой серии, а не гарантия валидности на реальных данных.','',
           f"Фильтр выполнен в {success_counts['sign_mom']} из 28 линейных сочетаний сценария/эффекта для исходных знаков + MoM, в {success_counts['sign_covariance']} из 28 для знаков + ковариации и в {success_counts['reference_transfer_fieller_hull']} из 28 для эталонного переноса. В эти знаменатели намеренно включены заранее заданные нарушения предпосылок и отсутствие конфаундинга. Эти доли не являются вероятностями успеха метода в природе.",'',
           '## Линейная модель: истинный эффект 0','',
           'В ячейках: RMSE / покрытие 95%-интервала. При истинном нуле ошибка I рода равна 1−покрытие.','',
           '| Сценарий | Без коррекции | Знаки + MoM | Знаки + ковариация | Перенос + Fieller |',
           '|---|---:|---:|---:|---:|']
    for s in names:
        cells=[]
        for m in ['uncalibrated','sign_mom','sign_covariance','reference_transfer_fieller_hull']:
            r=find(linear,s,m)
            cells.append(f"{num(r,'rmse'):.3f} / {num(r,'coverage'):.3f}")
        lines.append('| '+labels[s]+' | '+' | '.join(cells)+' |')
    lines+=['','## Восстановление ненулевого эффекта 0.1: линейная модель','',
            '| Сценарий | Процедура | Средняя оценка | Покрытие | Мощность | Ширина конечных интервалов | Неограниченные интервалы |',
            '|---|---|---:|---:|---:|---:|---:|']
    for s in ['equal_loadings','double_loadings','changed_units','two_factors_aligned','reference_shift','weak_reference']:
        for m,label in methods:
            r=find(linear,s,m,False)
            lines.append(f"| {labels[s]} | {label} | {.1+num(r,'bias'):+.3f} | {num(r,'coverage'):.3f} | {num(r,'rejection_rate'):.3f} | {num(r,'mean_finite_width'):.3f} | {num(r,'unbounded_fraction'):.3f} |")
    lines+=['','## Бинарный исход и cross-fit AIPW','',
            'В этой серии X и шумный прокси g включены в nuisance-модели; U скрыты. Использованы 2 складки cross-fitting, 500 повторений на строку сценария/эффекта и n=2000. Истинный ATE получен интегрированием потенциальных вероятностей, а не приравнен к коэффициенту логистической модели. Эталонный перенос здесь не проверялся.','',
            '| Сценарий | Истинный ATE | Процедура | Смещение ± MCSE | RMSE | Покрытие | Ошибка I рода / мощность | Успех по фильтру |',
            '|---|---:|---|---:|---:|---:|---:|---|']
    bnames=list(dict.fromkeys(r['scenario'] for r in binary))
    for s in bnames:
        for zero in [True,False]:
            for m,label in [('uncalibrated','Без коррекции'),('sign_mom','Знаки + MoM'),('sign_covariance','Знаки + ковариация')]:
                r=find(binary,s,m,zero)
                lines.append(f"| {labels[s]} | {num(r,'true_ate'):.4f} | {label} | {num(r,'bias'):+.4f} ± {num(r,'bias_mcse'):.4f} | {num(r,'rmse'):.4f} | {num(r,'coverage'):.3f} | {num(r,'rejection_rate'):.3f} | {'да' if success(r) else 'нет'} |")
    lines+=['','## Как интерпретировать результаты','',
            '1. **Эффективность исходного метода ограничена моделью общего смещения.** Уменьшение ошибки в согласованных сценариях является положительным свидетельством именно при этих условиях. Неуспех при изменении силы контролей сохраняет силу замечания редактора.','',
            '2. **Учитывать ковариацию важно.** Все контролируемые исходы и целевой исход измеряются на одних людях. Межконтрольная неоднородность в MoM и стандартная ошибка средней коррекции — разные величины. Вариант sign_covariance проверяет одно улучшение интервалов; он не исправляет несопоставимые величины смещения.','',
            '3. **Коэффициент переноса требует информации.** В линейной модели он оценён как отношение ковариаций с g в независимом эталоне: r=Cov(Y0,g)/Cov(Cbar,g). Это отношение вычисляется по данным эталона, без подстановки истинных нагрузок генератора. При одном общем факторе и инвариантных нагрузках оно переводит смещение контроля в масштаб исхода. При нескольких факторах нужна пропорциональность соответствующих связей, иначе отношение к U1 не переносит суммарное смещение.','',
            '4. **Эталон — сильное дополнительное условие.** Y0 должен быть сопоставимым исходом без воздействия в независимой репрезентативной выборке. Использовать просто наблюдаемых нелеченных студентов из конфаундированной основной выборки в качестве такого эталона без дополнительного обоснования нельзя. Эталонные данные в этом эксперименте симулированы; для OULAD/XES они не найдены.','',
            '5. **Слабая идентификация должна оставаться видимой.** При слабом знаменателе дельта-интервал может быть ненадёжен. Fieller-hull сохраняет неограниченные случаи, а не удаляет их. Столбец ширины относится только к конечным интервалам и читается вместе с долей неограниченных. Это асимптотическая инверсия, а не точный интервал.','',
            '6. **Знаки тоже оцениваются.** В вариантах с ковариацией, дельта-методом и Fieller выбранные знаки локально считаются фиксированными. Сильная неопределённость знаков, особенно в weak_reference, не полностью отражается в этих формулах. Поэтому наличие Fieller-hull не даёт общей гарантии покрытия при слабом прокси.','',
            '7. **Смена единиц и изменение информации различаются.** Общая смена единиц всех контролей оставляет эталонную коррекцию неизменной, что проверено алгебраически. Изменение единиц одного контроля меняет его вес в простом пуле и конечновыборочную дисперсию; при одном общем факторе отношение ковариаций по-прежнему компенсирует масштаб среднего смещения.','',
            '8. **Мощность сравнивается при корректном контроле ошибки.** Высокая частота значимых результатов у смещённой исходной оценки не доказывает хорошую мощность. Все числа мощности следует читать вместе с покрытием и результатами при нулевом эффекте. При 1000 повторениях MCSE доли 0.95 около 0.0069, при 500 — около 0.0097.','',
            '## Что можно утверждать в переработанной статье','',
            'Допустимая формулировка: «В заданных симуляционных моделях ориентация отрицательных контролей уменьшала ошибку при сопоставимости ориентированных смещений. Несопоставимые величины и неполные прокси приводили к нарушениям. Дополнительная эталонная информация позволяла оценить коэффициент переноса в линейной модели при инвариантности нагрузок; переносимость и неопределённость коэффициента являются существенными условиями».','',
            'Необоснованная формулировка: «Эксперименты доказали эффективность исходного метода для произвольных образовательных данных». Данная серия не валидирует реальные демографические контроли для retention и не устанавливает новизну отношения ковариаций или Fieller-интервалов. Литературное сравнение, анализ идентифицируемости и валидация на реальных данных остаются отдельными необходимыми этапами.','',
            '## Файлы и проверка','',
            '- PROTOCOL_RU.md и BINARY_PROTOCOL_RU.md: условия и критерии до запусков.',
            '- run_validation.py и run_binary_validation.py: расчётные скрипты (NumPy).',
            '- results/summary.csv и binary_results/summary.csv: все сводные числа, MCSE и критерии успеха.',
            '- В обеих папках results: отдельные повторения, конфигурации и контрольные суммы.',
            '- results/diagnostics.csv: коэффициенты переноса и диагностика слабого знаменателя для каждого линейного повторения.',
            '- Проверены алгебра интервала Fieller, логистические score-условия, квадратура истинного ATE, инвариантность к общему масштабу, сохранение добавленного линейного эффекта, нулевое смещение oracle-регрессии в пределах 5 MCSE и контрольные суммы результатов.','',
            'Воспроизведение из корня проекта:','',
            '```powershell',
            'rtk proxy python effectiveness_validation/run_validation.py',
            'rtk proxy python effectiveness_validation/run_binary_validation.py',
            'rtk proxy python effectiveness_validation/make_report.py',
            '```','']
    (ROOT/'REPORT_RU.md').write_text('\n'.join(lines),encoding='utf-8')
    audit=dict(linear_rows=len(linear),binary_rows=len(binary),checks='passed',linear_conditional_success=success_counts,
               binary_conditional_success={m:sum(success(r) for r in binary if r['method']==m) for m in ['sign_mom','sign_covariance']})
    (ROOT/'report_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))
    for rows,s,m in [(linear,'double_loadings','reference_transfer_fieller_hull'),
                      (linear,'equal_loadings','sign_covariance'),
                      (binary,'matched','sign_mom'),(binary,'matched','sign_covariance')]:
        r=find(rows,s,m)
        print(s,m,{key:r[key] for key in ['bias','rmse','coverage','rejection_rate']})


if __name__=='__main__': main()
