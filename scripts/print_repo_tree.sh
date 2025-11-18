#!/usr/bin/env bash
set -e
echo "Estrutura do repositório:"
command -v tree >/dev/null 2>&1 && tree -a -I "__pycache__|*.pyc" || find . -maxdepth 3 -print
