# Intrinio Python SDK for Real-Time Stock, Forex, and Crypto Prices

[Intrinio](https://intrinio.com/) provides real-time stock, forex, and crypto prices via a two-way WebSocket connection. To get started, [subscribe to a real-time data feed](https://intrinio.com/marketplace/data/prices/realtime) and follow the instructions below.

## Requirements

- Python 3.6

## Features

* Receive streaming, real-time price quotes (last trade, bid, ask)
* Subscribe to updates from individual securities, forex pairs, or cryptos
* Subscribe to updates for all securities, forex pairs, or cryptos (contact us for special access)

## Installation
```
pip install intriniorealtime
```

## Example Usage
```python
from intriniorealtime.client import IntrinioRealtimeClient

def on_quote(quote, backlog):
    print("QUOTE: " , quote, "BACKLOG LENGTH: ", backlog)
    
options = {
    'api_key': 'YOUR_INTRINIO_API_KEY',
    'provider': 'iex',
    'on_quote': on_quote
}

client = IntrinioRealtimeClient(options)
client.join(['AAPL','GE','MSFT'])
client.connect()
client.keep_alive()
```

## Handling Quotes and the Queue

When the Intrinio Realtime library receives quotes from the websocket connection, it places them in a queue. This queue has a default maximum size of 10,000 quotes. You can modify this value by specifying a `max_queue_size` option, as your environment memory constraints allow. Once a quote has been placed in the queue, it is passed to the `on_quote` handler in the order it was received. Your `on_quote` handler should process quotes quickly, so that the queue does not reach its maximum size (at which point, the system will log an error and ignore any incoming quotes until the queue has space). We recommend using the a thread class or pool for operations such as writing quotes to a database (or anything else involving time-consuming I/O). The `on_quote` handler also receives a `backlog` argument, which is an integer specifying the approximate length of the quote queue. Monitor this to make sure you are processing quotes quickly enough.

## Providers

Currently, Intrinio offers realtime data for this SDK from the following providers:

* IEX - [Homepage](https://iextrading.com/)
* QUODD - [Homepage](http://home.quodd.com/)
* Cryptoquote - [Homepage](https://www.cryptoquote.io/)
* FXCM - [Homepage](https://www.fxcm.com/)

Each has distinct price channels and quote formats, but a very similar API.

## Quote Data Format

Each data provider has a different format for their quote data.

### QUODD

NOTE: Messages from QUOOD reflect _changes_ in market data. Not all fields will be present in every message. Upon subscribing to a channel, you will receive one quote and one trade message containing all fields of the latest data available.

#### Trade Message

```python
{ 'ticker': 'AAPL.NB',
  'root_ticker': 'AAPL',
  'protocol_id': 301,
  'last_price_4d': 1594850,
  'trade_volume': 100,
  'trade_exchange': 't',
  'change_price_4d': 24950,
  'percent_change_4d': 15892,
  'trade_time': 1508165070052,
  'up_down': 'v',
  'vwap_4d': 1588482,
  'total_volume': 10209883,
  'day_high_4d': 1596600,
  'day_high_time': 1508164532269,
  'day_low_4d': 1576500,
  'day_low_time': 1508160605345,
  'prev_close_4d': 1569900,
  'volume_plus': 6333150,
  'ext_last_price_4d': 1579000,
  'ext_trade_volume': 100,
  'ext_trade_exchange': 't',
  'ext_change_price_4d': 9100,
  'ext_percent_change_4d': 5796,
  'ext_trade_time': 1508160600567,
  'ext_up_down': '-',
  'open_price_4d': 1582200,
  'open_volume': 100,
  'open_time': 1508141103583,
  'rtl': 30660,
  'is_halted': False,
  'is_short_restricted': False }
```

* **ticker** - Stock Symbol for the security
* **root_ticker** - Underlying symbol for a particular contract
* **last_price_4d** - The price at which the security most recently traded
* **trade_volume** - The number of shares that that were traded on the last trade
* **trade_exchange** - The market center where the last trade occurred
* **trade_time** - The time at which the security last traded in milliseconds
* **up_down** - Tick indicator - up or down - indicating if the last trade was up or down from the previous trade
* **change_price_4d** - The difference between the closing price of a security on the current trading day and the previous day's closing price.
* **percent_change_4d** - The percentage at which the security is up or down since the previous day's trading
* **total_volume** - The accumulated total amount of shares traded
* **volume_plus** - NASDAQ volume plus the volumes from other market centers to more accurately match composite volume. Used for NASDAQ Basic
* **vwap_4d** - Volume weighted Average Price. VWAP is calculated by adding up the dollars traded for every transaction (price multiplied by number of shares traded) and then dividing by the total shares traded for the day.
* **day_high_4d** - A security's intra-day high trading price.
* **day_high_time** - Time that the security reached a new high
* **day_low_4d** - A security's intra-day low trading price.
* **day_low_time** - Time that the security reached a new low
* **ext_last_price_4d** - Extended hours last price (pre or post market)
* **ext_trade_volume** - The amount of shares traded for a single extended hours trade
* **ext_trade_exchange** - Extended hours exchange where last trade took place (Pre or post market)
* **ext_trade_time** - Time of the extended hours trade in milliseconds
* **ext_up_down** - Extended hours tick indicator - up or down
* **ext_change_price_4d** - Extended hours change price (pre or post market)
* **ext_percent_change_4d** - Extended hours percent change (pre or post market)
* **is_halted** - A flag indicating that the stock is halted and not currently trading
* **is_short_restricted** - A flag indicating the stock is current short sale restricted - meaning you can not short sale the stock when true
* **open_price_4d** - The price at which a security first trades upon the opening of an exchange on a given trading day
* **open_time** - The time at which the security opened in milliseconds
* **open_volume** - The number of shares that that were traded on the opening trade
* **prev_close_4d** - The security's closing price on the preceding day of trading
* **protocol_id** - Internal Quodd ID defining Source of Data
* **rtl** - Record Transaction Level - number of records published that day

#### Quote Message

```python
{ 'ticker': 'AAPL.NB',
  'root_ticker': 'AAPL',
  'bid_size': 500,
  'ask_size': 600,
  'bid_price_4d': 1594800,
  'ask_price_4d': 1594900,
  'ask_exchange': 't',
  'bid_exchange': 't',
  'quote_time': 1508165070850,
  'protocol_id': 302,
  'rtl': 129739 }
```

* **ticker** - Stock Symbol for the security
* **root_ticker** - Underlying symbol for a particular contract
* **ask_price_4d** - The price a seller is willing to accept for a security
* **ask_size** - The amount of a security that a market maker is offering to sell at the ask price
* **ask_exchange** - The market center from which the ask is being quoted
* **bid_price_4d** - A bid price is the price a buyer is willing to pay for a security.
* **bid_size** - The bid size number of shares being offered for purchase at a specified bid price
* **bid_exchange** - The market center from which the bid is being quoted
* **quote_time** - Time of the quote in milliseconds
* **rtl** - Record Transaction Level - number of records published that day
* **protocol_id** - Internal Quodd ID defining Source of Data

### IEX

```python
{ 'type': 'ask',
  'timestamp': 1493409509.3932788,
  'ticker': 'GE',
  'size': 13750,
  'price': 28.97 }
```

*   **type** - the quote type
  *    **`last`** - represents the last traded price
  *    **`bid`** - represents the top-of-book bid price
  *    **`ask`** - represents the top-of-book ask price
*   **timestamp** - a Unix timestamp (with microsecond precision)
*   **ticker** - the ticker of the security
*   **size** - the size of the `last` trade, or total volume of orders at the top-of-book `bid` or `ask` price
*   **price** - the price in USD

### Cryptoquote

#### Level 1 - Price Update

NOTE: Nil values for some fields denote no change from previous value.

```python
{ 'last_updated': '2018-10-29 23:08:02.277Z',
  'pair_name': 'BTCUSD',
  'pair_code': 'btcusd',
  'exchange_name': 'Binance',
  'exchange_code': 'binance',
  'bid': 6326,
  'bid_size': 6.51933000,
  'ask': 6326.97,
  'ask_size': 6.12643000,
  'change': -151.6899999999996,
  'change_percent': -2.340895061728389,
  'volume': 13777.232772,
  'open': 6480,
  'high': 6505.01,
  'low': 6315,
  'last_trade_time': '2018-10-29 23:08:01.834Z',
  'last_trade_side': None,
  'last_trade_price': 6326.97000000,
  'last_trade_size': 0.00001200,
  'type': 'level_1' }
```

*   **last_updated** - a UTC timestamp of when the data was last updated
*   **pair_name** - the name of the currency pair
*   **pair_code** - the code of the currency pair
*   **exchange_name** - the name of the exchange
*   **exchange_code** - the code of the exchange
*   **ask** - the ask for the currency pair on the exchange
*   **ask_size** - the size of the ask for the currency pair on the exchange
*   **bid** - the bid for the currency pair on the exchange
*   **bid_size** - the size of the bid for the currency pair on the exchange
*   **change** - the notional change in price since the last ticker
*   **change_percent** - the percent change in price since the last ticker
*   **volume** - the volume of the currency pair on the exchange
*   **open** - the opening price of the currency pair on the exchange
*   **high** - the highest price of the currency pair on the exchange
*   **low** - the lowest price of the currency pair on the exchange
*   **last_trade_time** - a UTC timestamp of the last trade for the currency pair on the exchange
*   **last_trade_side** - the side of the last trade
  *    **`buy`** - this is an update to the buy side of the book
  *    **`sell`** - this is an update to the sell side of the book
*   **last_trade_price** - the price of the last trade for the currency pair on the exchange
*   **last_trade_size** - the size of the last trade for the currency pair on the exchange
*   **type** - the type of message this is
  *    **`level_1`** - a message that denotes a change to the last traded price or top-of-the-book bid or ask
  *    **`level_2`** - a message that denotes a change to an order book

#### Level 2 - Book Update

```python
{ 'pair_name': 'BTCUSD',
  'pair_code': 'btcusd',
  'exchange_name': 'Gemini',
  'exchange_code': 'gemini',
  'side': 'buy',
  'price': 6337.4,
  'size': 0.3,
  'type': 'book_update' }
```

*   **pair_name** - the name of the currency pair
*   **pair_code** - the code of the currency pair
*   **exchange_name** - the name of the exchange
*   **exchange_code** - the code of the exchange
*   **side** - the side of the book this update is for
  *    **`buy`** - this is an update to the buy side of the book
  *    **`sell`** - this is an update to the sell side of the book
*   **price** - the price of this book entry
*   **size** - the size of this book entry
*   **type** - the type of message this is
  *    **`level_1`** - a message that denotes a change to the last traded price or top-of-the-book bid or ask
  *    **`level_2`** - a message that denotes a change to an order book

### FXCM

#### Price update
```python
{ 'code': "EUR/USD",
  'bid_price': 1.13685,
  'ask_price': 1.13711,
  'time': "2018-12-18 22:38:06.964Z" }
```

*   **code** - the code of the fx currency pair
*   **bid_price** - the bid price is the price a buyer is willing to pay
*   **ask_price** - the ask price is the price a seller is willing to accept
*   **time** - the UTC timestamp of the price update

## Channels

### QUODD

To receive price quotes from QUODD, you need to instruct the client to "join" a channel. A channel can be
* A security ticker with data feed designation (`AAPL.NB`, `MSFT.NB`, `GE.NB`, etc)

### IEX

To receive price quotes from IEX, you need to instruct the client to "join" a channel. A channel can be
* A security ticker (`AAPL`, `MSFT`, `GE`, etc)
* The security lobby (`$lobby`) where all price quotes for all securities are posted
* The security last price lobby (`$lobby_last_price`) where only last price quotes for all securities are posted

Special access is required for both lobby channels. [Contact us](mailto:sales@intrinio.com) for more information.

### Cryptoquote

To receive price quotes from Cryptoquote, you need to instruct the client to "join" a channel. A channel can be

* `crypto:market_level_1:{pair_code}` - the Level 1 Market channel where all Level 1 price updates for the provided currency pair in all exchanges are posted (i.e. `crypto:market_level_1:btcusd`)
* `crypto:exchange_level_1:{exchange_code}:{pair_code}` - the Level 1 Market channel where all Level 1 price updates for the provided currency pair and exchange are posted
* `crypto:exchange_level_2:{exchange_code}:{pair_code}` - the Level 2 Market channel where all Level 2 book updates for the provided currency pair and exchange are posted
* `crypto:firehose` - the Firehose channel where all message types for all currency pairs are posted (special access required)

The Intrinio REST API provides a listing of pairs, exchanges, and their corresponding codes:

* [Crypto Currency Pairs](https://docs.intrinio.com/documentation/download/crypto_pairs)
* [Crypto Exchanges](https://docs.intrinio.com/documentation/download/crypto_exchanges)

### FXCM

To receive price quotes from FXCM, you need to instruct the client to "join" a channel. A channel can be

* `fxcm:pair:{pair_code}` - the fx currency pair channel where price updates for the provided currency pair are posted (i.e. `fxcm:pair:EUR/USD`)
* `fxcm:base:{currency_code}` - the fx currency channel where prices updates for any fx currency pair that has the provided currency as the base currency (i.e. `fxcm:base:EUR` would post prices from EUR/USD, EUR/JPY, etc.)
* `fxcm:quote:{currency_code}` - the fx currency channel where prices updates for any fx currency pair that has the provided currency as the quote currency (i.e. `fxcm:quote:USD` would post prices from EUR/USD, JPY/USD, etc.)

The Intrinio REST API provides a listing of pairs, currencies, and their corresponding codes:

* [FX Currencies](https://docs.intrinio.com/documentation/download/currencies)
* [FX Currency Pairs](https://docs.intrinio.com/documentation/download/currency_pairs)

## API Keys
You will receive your Intrinio API Key after [creating an account](https://intrinio.com/signup). You will need a subscription to a [realtime data feed](https://intrinio.com/marketplace/data/prices/realtime) as well.

## Documentation

### Methods

`client = IntrinioRealtimeClient(options)` - Creates an Intrinio Realtime client
* **Parameter** `options.api_key`: Your Intrinio API Key
* **Parameter** `options.provider`: The real-time data provider to use ("iex", "quodd", "cryptoquote")
* **Parameter** `options.on_quote(quote, backlog)`: (optional) A function that handles received quotes. `backlog` is an integer representing the approximate size of the queue of unhandled quotes
* **Parameter** `options.channels`: (optional) An array of channels to join after connecting
* **Parameter** `options.logger`: (optional) A Python Logger instance to use for logging

```python
def on_quote(quote, backlog):
    print("QUOTE: " , quote, "BACKLOG LENGTH: ", backlog)
    
options = {
    'api_key': 'YOUR_INTRINIO_API_KEY',
    'provider': 'iex',
    'on_quote': on_quote
}

client = IntrinioRealtimeClient(options)
```

---------

`client.connect()` - Retrieves an auth token, opens the WebSocket connection, starts the self-healing and heartbeat intervals, joins requested channels.

---------

`client.keep_alive()` - Runs an infinite loop to keep the thread alive, so that the client continues to receive prices. You may call this function after `connect()` or use your own timing logic (for example: connect, listen for quotes for x minutes, disconnect).

---------

`client.disconnect()` - Closes the WebSocket, stops the self-healing and heartbeat intervals. You _must_ call this to dispose of the client.

---------

`client.on_quote(quote, backlog)` - Changes the quote handler function
```python
def on_quote(quote, backlog):
    print("QUOTE: " , quote, "BACKLOG LENGTH: ", backlog)
    
client.on_quote = on_quote
```

---------

`client.join(channels)` - Joins the given channels. This can be called at any time. The client will automatically register joined channels and establish the proper subscriptions with the WebSocket connection.
* **Parameter** `channels` - A single channel or list of channels
```python
client.join(["AAPL", "MSFT", "GE"])
client.join("GOOG")
```

---------

`client.leave(channels)` - Leaves the given channels.
* **Parameter** `channels` - A single channel or list of channels
```python
client.leave(["AAPL", "MSFT", "GE"])
client.leave("GOOG")
```

---------

`client.leave_all()` - Leaves all channels.
