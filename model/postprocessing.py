from model.enums import Coin, CoinLabel
from model.coin_image import CoinImage
from typing import List


def get_total_count(coins: List[CoinImage]) -> int:
    total = 0
    for coin in coins:
        total += convert_label_to_coin(coin, coins).value
    return total / 100


def convert_label_to_coin(coin: CoinImage, coins: List[CoinImage]) -> Coin:
    if coin.label == CoinLabel.DUMMY:
        return Coin.DUMMY
    if (coin.label == CoinLabel.TEN_TWENTY_FIFTY_ONE_TAIL
            or coin.label == CoinLabel.ONE_TWO_FIVE_TAIL):
        return analyse_coin(coin, coins)
    return Coin[coin.label.name[:-5]]


"""TODO add toleration for radius"""
def analyse_coin(coin: CoinImage, coins: List[CoinImage]) -> Coin:
    similar_coins = list(filter(lambda x: is_similar_coin_known(coin, x), coins))
    if len(similar_coins) != 0:
        return Coin[similar_coins[0].label.name[:-5]]
    similar_class_coins = unique(list(sorted(filter(lambda x: is_similar_label_coin(coin, x), coins),
                                                 key=lambda x: x.radius)))
    return get_coin(coin, similar_class_coins)


"""TODO add toleration for radius"""
def is_similar_coin_known(coin: CoinImage, iter_coin: CoinImage) -> bool:
    if (coin.radius == iter_coin.radius
            and iter_coin.label != CoinLabel.ONE_TWO_FIVE_TAIL
            and iter_coin.label != CoinLabel.TEN_TWENTY_FIFTY_ONE_TAIL):
        return True
    return False


def is_similar_label_coin(coin: CoinImage, iter_coin: CoinImage) -> bool:
    if iter_coin.label == coin.label:
        return True
    return False


"""TODO add toleration for radius"""
def unique(coins: List[CoinImage]) -> List[CoinImage]:
    unique_coins = []
    for coin in coins:
        if coin not in unique_coins:
            unique_coins.append(coin)
    return unique_coins


"""TODO add toleration for radius"""
def get_coin(coin: CoinImage, similar_class_coins: List[CoinImage]) -> Coin:
    index = 0
    for i in range (0, len(similar_class_coins)):
        if coin.radius == similar_class_coins[i].radius:
            index = i
            break
    if coin.label == CoinLabel.ONE_TWO_FIVE_TAIL:
        return one_two_five_to_coin(index)
    return ten_twenty_fifty_one_to_coin(index)


def one_two_five_to_coin(index: int) -> Coin:
    if index == 0:
        return Coin.ONE_GROSZY
    if index == 1:
        return Coin.TWO_GROSZY
    return Coin.FIVE_GROSZY


def ten_twenty_fifty_one_to_coin(index: int) -> Coin:
    if index == 0:
        return Coin.TEN_GROSZY
    if index == 1:
        return Coin.TWENTY_GROSZY
    if index == 2:
        return Coin.FIFTY_GROSZY
    return Coin.ONE_ZLOTY
