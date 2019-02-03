from model.enums import CoinLabel


class CoinImage:
    def __init__(self, radius: int):
        self.radius = radius
        self.label = CoinLabel.DUMMY

    def __init__(self, radius: int, label: CoinLabel):
        self.radius = radius
        self.label = label

