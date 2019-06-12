import unittest
import model.preprocessing

class TestHoughTransform(unittest.TestCase):

    def test_test_coins_1(self):
        self.check_num_of_coins("test_coins_1.jpg", 4)

    def test_test_coins_2(self):
        self.check_num_of_coins("test_coins_2.jpg", 6)

    def test_test_coins_3(self):
        self.check_num_of_coins("test_coins_3.jpg", 7)

    def test_test_coins_4(self):
        self.check_num_of_coins("test_coins_4.jpg", 7)

    def test_test_coins_5(self):
        self.check_num_of_coins("test_coins_5.jpg", 9)

    def test_t1(self):
        self.check_num_of_coins("t1.jpg", 7)

    def test_t2(self):
        self.check_num_of_coins("t2.jpg", 7)

    def test_t3(self):
        self.check_num_of_coins("t3.jpg", 7)

    def test_t4(self):
        self.check_num_of_coins("t4.jpg", 4)

    def test_t5(self):
        self.check_num_of_coins("t5.jpg", 7)

    def test_t6(self):
        self.check_num_of_coins("t6.jpg", 7)

    def test_coins(self):
        self.check_num_of_coins("coins.jpg", 18)

    def check_num_of_coins(self, filename: str, num: int):
        path_prefix = "data/images_of_multiple_coins/"
        coin_images = model.preprocessing.hough_transform(path_prefix + filename)
        self.assertEqual(len(coin_images), num)


if __name__ == '__main__':
    unittest.main()
