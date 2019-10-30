//+------------------------------------------------------------------+
//|                                             download_dataset.mq5 |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"

input int from_year = 1984;
input int to_year = 2019;
input string symbol = "XAUUSD";
input string output = "dataset";
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
{
  for (int y = from_year ; y <= to_year ; y++) {
    for (int m = 1 ; m <= 12 ; m += 3) {
      // 3 month's ticks per file
      int from_month = m;
      int to_month = m + 2;
      string strTime = y + "/" + from_month + "/" + 1 + " " + 0 + ":" + 0 + ":" + 0;
      datetime from_datetime = StrToTime(strTime);
      int last_day = (to_month == 3||to_month == 12)?31:30;
      strTime = y + "/" + to_month + "/" + last_day + " " + 23 + ":" + 59 + ":" + 59;
      string filename = StringFormat("%s_%s_%s.csv", symbol, TimeToString(from_datetime), TimeToString(to_datetime));
      double ticks[];
      ZeroMemory(ticks);
      int copyTime = CopyTicksRange(symbol, ticks, COPY_TICKS_ALL, from_datetime, to_datetime);
      // write to file      
      int file = FileOpen(output + "\\" + filename, FILE_WRITE|FILE_COMMON);
      // TODO: save to file
    }
  }
}
//+------------------------------------------------------------------+
