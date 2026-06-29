# Локальный прогон mini-swe-agent на LM Studio (бесплатно, ноль трат)

*Рабочий рецепт от 2026-06-21. Подтверждено: агент решает задачу end-to-end на локальной модели, без API-ключей и без трат.*

## WSL на хосте (актуально, 2026-06-23)
Сейчас работаем из WSL2, не из VM/Windows. Рецепт ниже (PowerShell) — историческая Windows-версия. Для WSL всё вынесено в [env-wsl.sh](env-wsl.sh): `source env-wsl.sh` подставит base/key/model и сам выведет хост-IP. Отличия от Windows-рецепта:
- Хост-IP ≠ `localhost` (WSL = NAT): берётся из `ip route show default` (`<lm-studio-host>` на 2026-06-23). env-wsl.sh делает это динамически.
- LM Studio сейчас без auth → `LM_STUDIO_API_KEY` = плейсхолдер `lm-studio`. Включат permission token — вписать реальный (секрет, не коммитить).
- `LocalEnvironment` на Linux = bash → грабля №3 (`cmd.exe`/`dir`) только для Windows; `ls` тут работает.
Запуск: `cd ~/projects/my-unsloth-finetune && source env-wsl.sh && "$MSWEA_PY" local_smoke.py "задача"`.

## Топология
- **Хост (DON PERIGNON):** LM Studio, сервер на `0.0.0.0:1234`, модели: `qwen2.5-coder-7b-instruct`, `mistralai/ministral-3-14b-reasoning`, `text-embedding-nomic-embed-text-v1.5`. Адрес, видимый из VM: **`http://10.2.0.2:1234`** (не gateway 172.18.64.1 – у LM Studio свой адрес, смотреть в логе сервера «Reachable at»).
- **VM (AICLAW):** mini-swe-agent + venv (Python 3.13). Отсюда гоняем харнесс, модель – по сети на хосте.
- Авторизация: permission token LM Studio лежит в VM как `LM_STUDIO_API_KEY` (через `setx`, в чат не пишем).

## Рабочая команда
Из `mini-swe-agent/` (или из любой рабочей папки – агент пишет в текущую cwd):
```
$env:PYTHONUTF8 = "1"
$env:LM_STUDIO_API_KEY = [Environment]::GetEnvironmentVariable('LM_STUDIO_API_KEY','User')
$env:LM_STUDIO_API_BASE = "http://10.2.0.2:1234/v1"
$env:MSWEA_COST_TRACKING = "ignore_errors"
$env:MSWEA_MODEL_NAME = "lm_studio/qwen2.5-coder-7b-instruct"
& "<...>\mini-swe-agent\.venv\Scripts\python.exe" "<...>\local_smoke.py" "твоя задача"
```
Раннер: [local_smoke.py](local_smoke.py). Успех = `RUN RESULT: {'exit_status': 'Submitted', ...}` + ожидаемый артефакт в рабочей папке.

## Три грабли, на которые наступили (это и есть harness-инженерия)
1. **Локальные модели ≠ native tool-calls.** Дефолтный `LitellmModel` (v2) жёстко шлёт `tools=[BASH_TOOL]` и читает только `message.tool_calls`. qwen в LM Studio tool-calls не отдаёт → три «No tool calls found» подряд → `RepeatedFormatError`.
   **Фикс:** `LitellmTextbasedModel` – он НЕ шлёт tools и парсит блок ` ```mswea_bash_command ``` из текста регуляркой. `hello_world.py` его не использует, поэтому написан свой раннер.
2. **Cost-tracking падает на локалке.** litellm не знает цены `lm_studio/...` → `RuntimeError: This model isn't mapped yet`.
   **Фикс:** `MSWEA_COST_TRACKING=ignore_errors` (для бесплатной модели считать нечего).
3. **Windows-консоль.** Баннер пакета содержит эмодзи → краш на cp1252.
   **Фикс:** `PYTHONUTF8=1`. И помни: `LocalEnvironment` исполняет команды через `cmd.exe`, не bash – `ls` не работает (агент сам переключился на `dir`).

## Что увидели (verify-loop вживую)
Агент выдал `echo "HELLO WORLD" > hello.txt && ls -la` → среда вернула «'ls' is not recognized» → агент прочитал ошибку, переключился на `dir /b` → проверил → завершил `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. Самокоррекция после ошибки среды – это ровно тот компонент харнесса (verification loop), который мы изучаем.

## Безопасность
`LocalEnvironment` исполняет команды модели **прямо в VM, без песочницы**. Для обучения на тривиальных задачах ок; для прогонов с непредсказуемыми задачами (SWE-bench и т.п.) – переключаться на Docker-окружение (Docker в VM есть) или на хостовый Docker. (Жюри ничего НЕ исполняет – судьи только читают текст и ставят баллы, песочница не нужна.)

## Жюри FreeStack (verdict → прямой litellm, 2026-06-23)
Раннер: [jury/jury.py](jury/jury.py), отдельный venv `jury/.venv`. Запуск:
```
cd ~/projects/my-unsloth-finetune && source env-wsl.sh && jury/.venv/bin/python jury/jury.py
```
Успех = строка `DISCRIMINATION: ... PASS` + файл `jury/runs/verdict_*.json`. Судьи (разные семейства, агент-кодер НЕ судья) переопределяются через `JURY_JUDGES="lm_studio/...,lm_studio/..."`. Гоняются по очереди (VRAM 12 ГБ), judge-внешний цикл → модель грузится один раз.

### Грабли №4–5 (verdict + локальные модели — почему ушли на прямой litellm)
4. **verdict ≠ LM Studio из коробки.** verdict 0.2.7 патчит litellm через instructor в `Mode.TOOLS` → шлёт `tool_choice` ОБЪЕКТОМ → LM Studio `400 Invalid tool_choice type: 'object'`. Форс `Mode.JSON` → `response_format=json_object` → LM Studio `400 'response_format.type' must be 'json_schema' or 'text'`. **LM Studio принимает только `response_format=json_schema`** (грамматика-констрейн структурного вывода).
5. **instructor 1.7.2 ретрай-баг.** verdict хард-пинит `instructor==1.7.2`; его ретрай-обёртка падает против `tenacity` 9.x (`Retrying.__call__() got multiple values for 'self'`) на КАЖДОМ вызове, маскируя настоящую ошибку.
**Фикс:** обойти verdict — прямой `litellm.completion(model="lm_studio/<id>", response_format={"type":"json_schema","json_schema":{...}}, temperature=0)`. Проверено: phi-4 / ministral / llama дают валидный JSON; жюри различает корректный ответ (5) и баг (1) единогласно. Деталь: reasoning-судьи (ministral, deepseek-r1) под json_schema-грамматикой не выдают `<think>` (констрейн заставляет JSON сразу) – для оценки ок, но CoT подавлен.
