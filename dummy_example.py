
# This is a pattern I guess useful for switching 1 lines in inherites codes
class Child():

    def __init__(self):
        self.money = 10
        print(self.manipulation(self.money))

    def manipulation(self,money):
        pass


class SpoiledChild(Child):

    def __init__(self):
        super().__init__()

    def manipulation(self, money):
        return money*10


if __name__ == '__main__':

    father = SpoiledChild()

