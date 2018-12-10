import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import requests as r
import seaborn as sns


# To use this, create a file called 'cryptocompare_apikey' in the research
# directory and it should just contain {"APIKEY": "(INSERT API KEY HERE)"}
APIKEY = json.loads(open("../cryptocompare_apikey.json").read())["APIKEY"]

BASEURL = "https://min-api.cryptocompare.com/data/"


def load_historical_cryptodata(fsym, tsym, time_end, freq, nobs):
    if not os.path.exists("../data"):
        os.mkdir("../data")

    get_url = BASEURL + "histo" + {"D": "day", "H": "hour", "M": "minute"}[freq]

    _params = {
        "fsym": fsym, "tsym": tsym,
        "toTs": int(pd.to_datetime(time_end).timestamp()),
        "limit": nobs,
        "api_key": APIKEY
    }

    file_hash = hash(get_url + str(_params))
    fn = "cryptocompare_{}".format(file_hash)

    if os.path.exists("../data/" + fn):
        df = pd.read_csv("../data/" + fn, parse_dates=["time"]).set_index("time")
    else:
        the_request = r.get(get_url, params=_params)
        df = pd.DataFrame(json.loads(the_request.text)["Data"])
        df["time"] = pd.to_datetime(df["time"], unit="s")

        # Save file
        df.to_csv("../data/" + fn)
        df = df.set_index("time")

    return df


def load_realtime_cryptodata(fsyms, tsyms):
    get_url = BASEURL + "pricemultifull"

    _params = {"fsyms": fsyms, "tsyms": tsyms, "api_key": APIKEY}
    the_request = r.get(get_url, params=_params)
    data = json.loads(the_request.text)["RAW"]

    datacols = [
        "PRICE", "CHANGE24HOUR"
    ]

    the_index = fsyms.split(",")
    the_cols = []
    for _sym in tsym.split(","):
        for _data in datacols:
            the_cols.append((_sym, _data))

    df = pd.DataFrame(columns=pd.MultiIndex.from_tuples(the_cols), index=the_index)
    for _fsym in the_index:
        for (_tsym, _col) in the_cols:
            df.at[_fsym, (_tsym, _col)] = data[_fsym][_tsym][_col]

    return df


df = load_historical_cryptodata("ETH", "USD", "12/4/2018", "H", 120)

df.plot(y="open")
