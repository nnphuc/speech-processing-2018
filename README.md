# speech-processing-2018
Dự án môn học xử lý tiếng nói

## Mục đích:
- Dùng để phiên dịch câu tiếng Anh sang tiếng Việt
- Tương lai: huấn luyện mô hình ngược lại, đọc kết quả, đưa lên điện thoại

# Mô hình sử dụng:  [Transformer](https://arxiv.org/abs/1706.03762)


## Cấu trúc thư mục:

```
speech-processing-2018:
|   dataset.py
│   ex.txt
│   LICENSE
│   model.chkpt
│   multi-bleu.perl
│   pred.txt
│   preprocess.py
│   README.md
│   sr.py
│   tokenizer.perl
│   train.py
│   translate-gui.py
│   _gitignore
│
├───.idea
│       misc.xml
│       modules.xml
│       speech-processing-2018.iml
│       vcs.xml
│       workspace.xml
│
├───.pytest_cache
│   │   .gitignore
│   │   CACHEDIR.TAG
│   │   README.md
│   │
│   └───v
│       └───cache
│               nodeids
│               stepwise
│
├───data
│       dict.pt
│
├───transformer
│   │   Beam.py
│   │   Constants.py
│   │   Layers.py
│   │   Models.py
│   │   Modules.py
│   │   Optim.py
│   │   SubLayers.py
│   │   Translator.py
│   │   __init__.py
│   │

```

## Huấn luyện mô hình:
*Tham số mô hình*

| Tên        | Giá trị           | 
| ------------- |:-------------:| 
| embedding size      | 100 | 
| n_head    | 8      |  
| n_layer | 6      |  
| d_inner | 1024 |
| batch size | 64 |

## Dữ liệu:
 [https://drive.google.com/drive/folders/0BzY0S4QyX701VU5LdVY4X3BzbVE](https://drive.google.com/drive/folders/0BzY0S4QyX701VU5LdVY4X3BzbVE)
