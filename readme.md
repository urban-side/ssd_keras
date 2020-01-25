Traffic survey with SSD_keras as academic research.
====

## Description
実際に卒研で使ったコードの可読性・使いやすさを向上させたもの。[rykov8氏のSDをkerasで実装したプログラム](https://github.com/rykov8/ssd_keras)を元にした。ライセンスも彼に準拠する。

## Requirement setup
Please type `pip install -r requirement.txt` on here. Required libraries will be automatically installed.

## Demo
### `pic_prediction_demo.py`:
Demo with several images. Pics path is `dataset/test_pics/`.

### `videotest_demo.py`:
Demo with video or camera input. If use webcam remove the parameter in line: 24.
>`vid_test.run(VIDEO_PATH)` > `vid_test.run()`

## Train
Type `$ mkdir dataset/labels/ dataset/pics/`. Datasets use these directory. Labels format is `*.xml`, click [here](https://qiita.com/slowsingle/items/64cc927bb29a49a7af14) for reference.

After make `*.pkl`, fetch paths in `ssd_training.py` and run. Later `*.hdf5` outputs.

## Licence

[MIT LICENCE](https://github.com/tcnksm/tool/blob/master/LICENCE)

[Reference source LICENCE](https://github.com/rykov8/ssd_keras/blob/master/LICENSE)
