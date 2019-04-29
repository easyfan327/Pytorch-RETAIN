import os

if __name__ == "__main__":
    for i in range(30):
        print("test {}".format(i))
        os.system("python main.py >> {}.csv".format(i))