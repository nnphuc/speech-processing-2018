''' Translate input text with trained model. '''

import torch
import torch.utils.data
from dataset import collate_fn, TranslationDataset
from transformer.Translator import Translator
from preprocess import read_instances_from_sent, convert_instance_to_idx_seq
import speech_recognition
import nltk


class Opt:
    vocab = "data/dict.pt"
    batch_size = 1
    cuda = True
    model = "model.chkpt"
    beam_size = 5
    n_best = 1


opt = Opt()
preprocess_data = torch.load(opt.vocab)
preprocess_settings = preprocess_data['settings']
sent = "hello , how are you ?"

translator = Translator(opt)


def translate(sent):
    input_word_insts = read_instances_from_sent(
        sent,
        preprocess_settings.max_word_seq_len,
        preprocess_settings.keep_case)
    test_src_insts = convert_instance_to_idx_seq(
        input_word_insts, preprocess_data['dict']['src'])

    test_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=preprocess_data['dict']['src'],
            tgt_word2idx=preprocess_data['dict']['tgt'],
            src_insts=test_src_insts),
        batch_size=opt.batch_size,
        collate_fn=collate_fn)

    for batch in test_loader:
        all_hyp, all_scores = translator.translate_batch(*batch)

        for src_seqs, idx_seqs in zip(batch[0].numpy(), all_hyp):

            for idx_seq in idx_seqs:
                pred_line = ' '.join([test_loader.dataset.tgt_idx2word[idx] for idx in idx_seq])
                print(pred_line)


import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QDesktopWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 button - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 320
        self.height = 200
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)

        centerPoint = QDesktopWidget().availableGeometry().center()
        self.move(centerPoint)
        button = QPushButton('PyQt5 button', self)
        button.setToolTip('This is an example button')
        button.move(100, 70)
        button.clicked.connect(self.on_click)

        self.show()

    @pyqtSlot()
    def on_click(self):
        print('PyQt5 button click')
        import speech_recognition as sr

        # obtain audio from the microphone
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Please wait. Calibrating microphone...")
            # listen for 5 seconds and create the ambient noise energy level
            r.adjust_for_ambient_noise(source, duration=5)
            print("Say something!")
            audio = r.listen(source)

            # recognize speech using Sphinx
        try:
            print("Sphinx thinks you said '" + r.recognize_sphinx(audio) + "'")
        except sr.UnknownValueError:
            print("Sphinx could not understand audio")
        except sr.RequestError as e:
            print("Sphinx error; {0}".format(e))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
