from OpticalRecognizer import OpticalRecognizer
from ImageProcessor import ImageProcessor
from ImageCapture import ImageCapture

class Recognizer:

    optRecognizer: OpticalRecognizer
    imgProcessor: ImageProcessor

    def __init__(self, lang: str):
        self.optRecognizer = OpticalRecognizer(lang)
        self.imgProcessor = ImageProcessor()

    def getLanguage(self) -> str:
        return self.optRecognizer.getLanguage()

    def getOptRecognizer(self) -> OpticalRecognizer:
        return self.optRecognizer

    def getImgProcessor(self) -> ImageProcessor:
        return self.imgProcessor

    def setLanguage(self, lang: str) -> None:
        self.optRecognizer.setLanguage(lang)

    def setOptRecognizer(self, optRecognizer: OpticalRecognizer) -> None:
        self.optRecognizer = optRecognizer

    def setImgProcessor(self, imgProcessor: ImageProcessor) -> None:
        self.imgProcessor = imgProcessor
