# Локальный v1.1 → GitHub

Подготовлен комплект **только для актуальной Physica A** в ветке `sep_research`. Commit, tag, push и публикация не выполнялись. [README](<README.md>) описывает содержимое, [заметки релиза](<RELEASE_NOTES.md>) — ограничения, [manifest](<RELEASE_MANIFEST.json>) — размеры и SHA-256.

Из индекса убраны 316 старых файлов, не относящихся к этому релизу; **их локальные копии сохранены**. Исходные локальные изменения также сохранены отдельно. История Git не переписывалась: исключения относятся к новому дереву и ZIP, а не к прошлым коммитам. Рабочая папка параллельной подготовки статьи не изменялась.

## 1. Просмотр и локальный commit/tag

Из корня репозитория:

```sh
set -e
test "$(git branch --show-current)" = sep_research
git status --short
git diff --cached --stat
git add --pathspec-from-file=RELEASE_FILES.txt
git diff --cached --stat
```

Просмотрите изменения. Не используйте `git add -A`: локальные рабочие материалы не должны попасть в релиз. [RELEASE_FILES.txt](<RELEASE_FILES.txt>) — точный разрешённый состав. Проверить, свободен ли тег на GitHub:

```sh
git ls-remote --tags origin refs/tags/v1.1
```

Если вывод не пустой, остановитесь и согласуйте другой тег; не перезаписывайте существующий. Затем:

```sh
git commit -m "Release v1.1: Physica A reproduction materials"
git tag -a v1.1 -m "Physica A reproduction materials v1.1"
```

## 2. Загрузка ветки и тега

```sh
git push -u origin sep_research
git push origin refs/tags/v1.1
```

## 3. ZIP из тега и GitHub Release

```sh
test ! -e ../traffic_rl-v1.1.zip
git archive --format=zip --prefix=traffic_rl-v1.1/ \
  --output=../traffic_rl-v1.1.zip v1.1
gh release create v1.1 ../traffic_rl-v1.1.zip --verify-tag \
  --title "v1.1 — Physica A" --notes-file RELEASE_NOTES.md
```

Для последней команды нужен установленный и авторизованный GitHub CLI (`gh auth login`). Альтернатива — создать Release в веб-интерфейсе GitHub для уже загруженного тега `v1.1` и приложить этот ZIP.

Проверки подготовки были только статическими и файловыми; тесты, исследования и перерисовка не запускались. Файлы этой версии меньше лимита GitHub в 100 MiB; Git LFS для нового дерева не требуется.
