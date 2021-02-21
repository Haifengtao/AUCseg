import pandas as pd


class Logger_csv(object):
    def __init__(self, title, out_dir):
        """
        save the msg during iteration.
        :param title: table's title;
        :param out_dir: output path;
        """
        self.dict = {}
        self.out_dir = out_dir
        for i in title:
            self.dict[i] = []

    def update(self, msg):
        try:
            for i in self.dict.keys():
                self.dict[i].append(msg[i])
        except Exception as err:
            print(err)
        df = pd.DataFrame(self.dict)
        df.to_csv(self.out_dir)


if __name__ == '__main__':
    a = ['1','wq']
    logger = Logger_csv(a,'./test.csv')
    x={"1":2121,'wq':32}
    logger.update(x)
