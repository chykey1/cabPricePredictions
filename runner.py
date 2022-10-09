import pandas as pd


def runner():

    data_weather = pd.read_csv("data/weather.csv")
    data_rides = pd.read_csv("data/weather.csv")

    return data_weather, data_rides


if __name__ == "__main__":

    weather, rides = runner()
