from model.enums import CoinLabel


class CoinImage:
    def __init__(self, radius: int, label: CoinLabel = CoinLabel.DUMMY):
        self.radius = radius
        self.label = label
