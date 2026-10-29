# 统一运行环境

A在虚拟环境中安装根目录requirements.txt，再运行：

```bash
python --version > environment/python-version.txt
python -m pip freeze > environment/requirements-lock.txt
```

记录操作系统；B先确认此环境可运行，再冻结版本。B/C使用相同Python主次版本及锁定的依赖。若必须更新依赖，应由组长记录原因，并重新核对受影响结果。不要上传.venv或venv目录。
