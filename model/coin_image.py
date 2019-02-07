from model.enums import CoinLabel, Coin
import numpy as np


class CoinImage:
    def __init__(self, radius: int, label: CoinLabel = CoinLabel.DUMMY, coin: Coin = Coin.DUMMY, img_arr: np.ndarray = np.zeros(shape=(1, 1))):
        self.radius = radius
        self.label = label
        self.coin = coin
        self.img_arr = img_arr
