import numpy as np

from model.enums import CoinLabel, Coin


class CoinImage:
    def __init__(self, radius: int = 0, label: CoinLabel = CoinLabel.DUMMY, coin: Coin = Coin.DUMMY,
                 img_arr: np.ndarray = np.zeros(shape=(1, 1))):
        self.radius = radius
        self.label = label
        self.coin = coin
        self.img_arr = img_arr
