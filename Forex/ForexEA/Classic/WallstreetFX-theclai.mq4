
#property copyright "Copyright � theclai"




extern string Name = "WallStreet Forex Robot modified by theclai";
extern string Copy = "Copyright �theclai";
extern string Op2 = "";
extern string Symbol_Op = "EURUSD m15";
extern string Op = "";
extern datetime Date = D'12.05.2011'; //--- ������ ����� ��� ����, ����� ������ �� ����� ���� ����������� (���� ���������� �������)
extern string _TP = "";
//---
extern int TakeProfit = 15; //--- (10 2 60)
extern int StopLoss = 120; //--- (100 10 200)
extern bool UseStopLevels = TRUE; //--- ��������� �������� �������. ���� ���������, �� �������� ������ ����������� ����� � �����.
extern bool IsMarketExecution = true; //--- ��������� ��������� ���������� �������� ������� (������� ���������, ����� ������������)
//---
extern int SecureProfit = 5; //--- (0 1 5) ����� � ���������
extern int SecureProfitTriger =-10; //--- (10 2 30)
extern int MaxLossPoints = 3; //--- (-200 5 -20) ������������ �������� ��� �������� ������� Buy � Sell ��� ��������� ������� (��� �������� ������ �� - MaxLossPoints ��� ������ (�������� ������� 0), ����� ���������)

extern string _MM = "MM";
//---
extern bool RecoveryMode = true; //--- ��������� ������ �������������� �������� (���������� ���� ���� �������� ����-����)
extern double FixedLot = 0.01; //--- ������������� ����� ����
extern double AutoMM = 20.0; //--- �� ���������� ���� AutoMM > 0. ������� �����. ��� RecoveryMode = FALSE, ������ ����� ������ ��� ��������.
//--- ��� AutoMM = 20 � �������� � 1000$, ��� ����� ����� 0,2. ����� ��� ����� ������������� ������ �� ��������� �������, �� ���� ��� ��� �������� � 2000$ ��� ����� ����� 0,4.
extern double AutoMM_Max = 30.0; //--- ������������ ����
extern int MaxAnalizCount = 50; //--- ����� �������� ����� ������� ��� �������(������������ ��� RecoveryMode = True)
extern double Risk = 45.0; //--- ���� �� �������� (������������ ��� RecoveryMode = True)
extern double MultiLotPercent = 1.1; //--- ����������� ��������� ���� (������������ ��� RecoveryMode = True)

extern string _Vola = "";
// ������ �������������
extern int VolaFilter = 23; //--- (15 1 30)

//+--------------------------------------------------------------------------------------------------------------+
//| ������� �����������. ���-�� ����� ��� ������� ����������.
//+--------------------------------------------------------------------------------------------------------------+

extern string _Periods = "";

//--- ������� ����������� (���� ����� ����� ��������, ��� ��� ��� ������ ���� ����)
extern int iMA_Period = 75; //--- (60 5 100)
extern int iCCI_Period = 18; //--- (10 2 30)
extern int iATR_Period = 14; //--- (10 2 30) (!) ����� �� ������
extern int iWPR_Period = 11; //--- (10 1 20)

//+--------------------------------------------------------------------------------------------------------------+
//| ��������� �� DLL
//+--------------------------------------------------------------------------------------------------------------+
//| EURUSD     | GBPUSD     | USDCHF     | USDJPY     | USDCAD     |
//+----------------------------------------------------------------
//| TP=26;     | TP=50;     | TP=17;     | TP=27;     | TP=14;     |
//| SL=120;    | SL=120;    | SL=120;    | SL=130;    | SL=150;    |
//| SP=1;      | SP=2;      | SP=0;      | SP=2;      | SP=2;      |
//| SPT=10;    | SPT=24;    | SPT=15;    | SPT=14;    | SPT=10;    |
//| MLP=-65;   | MLP=-200;  | MLP=-40;   | MLP=-80;   | MLP=-30;   |
//+----------------------------------------------------------------
//| MA=75;     | MA=75;     | MA=70;     | MA=85;     | MA=65;     |
//| CCI=18;    | CCI=12;    | CCI=14;    | CCI=12;    | CCI=12;    |
//| ATR=14;    | ATR=14;    | ATR=14;    | ATR=14;    | ATR=14;    |
//| WPR=11;    | WPR=12;    | WPR=12;    | WPR=12;    | WPR=16;    |
//+----------------------------------------------------------------
//| FATR=6;    | FATR=6;    | FATR=3;    | FATR=0;    | FATR=4;    |
//| FCCI=150;  | FCCI=290;  | FCCI=170;  | FCCI=2000; | FCCI=130;  |
//+----------------------------------------------------------------
//| MAFOA=15;  | MAFOA=12;  | MAFOA=8;   | MAFOA=5;   | MAFOA=5;   |
//| MAFOB=39;  | MAFOB=33;  | MAFOB=25;  | MAFOB=21;  | MAFOB=15;  |
//| WPRFOA=-99;| WPRFOA=-99;| WPRFOA=-95;| WPRFOA=-99;| WPRFOA=-99;|
//| WPRFOB=-95;| WPRFOB=-94;| WPRFOB=-92;| WPRFOB=-95;| WPRFOB=-89;|
//+----------------------------------------------------------------
//| MAFC=14;   | MAFC=18;   | MAFC=11;   | MAFC=14;   | MAFC=14;   |
//| WPRFC=-19; | WPRFC=-19; | WPRFC=-22; | WPRFC=-27; | WPRFC=-6;  |
//+--------------------------------------------------------------------------------------------------------------+

//+--------------------------------------------------------------------------------------------------------------+
//| ��������� ����������� ��� ������ �������� � �������� �������.
//+--------------------------------------------------------------------------------------------------------------+
extern string _Add_Op = "";
//---
extern string _AddOpenFilters = "---";
//---
extern int FilterATR = 6; //--- (0 1 10) �������� �� ���� �� ATR ��� Buy � Sell (if (iATR_Signal <= FilterATR * pp) return (0);) (!) ����� �� ������
extern double iCCI_OpenFilter = 150; //--- (100 10 400) ������ �� iCCI ��� Buy � Sell. ��� ����������� ��� JPY ������������ ������ �� ������� (100 50 4000)
//---
extern string _OpenOrderFilters = "---";
//---
extern int iMA_Filter_Open_a = 15; //--- (4 2 20) ������ �� ��� �������� Buy � Sell (�����)
extern int iMA_Filter_Open_b = 39; //--- (14 2 50) ������ �� ��� �������� Buy � Sell (�����)
extern int iWPR_Filter_Open_a = -99; //--- (-100 1 0) ������ WPR ��� �������� Buy � Sell
extern int iWPR_Filter_Open_b = -95; //--- (-100 1 0) ������ WPR ��� �������� Buy � Sell
//---
extern string _CloseOrderFilters = "---";
//---
extern int Price_Filter_Close = 14; //--- (10 2 20) ������ ���� �������� ��� �������� Buy � Sell (�����)
extern int iWPR_Filter_Close = -19; //--- (0 1 -100) ������ WPR ��� �������� Buy � Sell

//+--------------------------------------------------------------------------------------------------------------+
//| ����������� ���������
//+--------------------------------------------------------------------------------------------------------------+

extern string _Add = "";
extern bool LongTrade = TRUE; //--- ����������� ������� �������
extern bool ShortTrade = TRUE; //--- ����������� �������� �������
extern bool No_Hedge_Trades = false; //--- ������������� �������� ������ Buy � ������ Sell ������. ��� True �� ���������.
extern bool OneOrderInBarMode = TRUE; //--- ��� True �������� ����� ��������� ������ 1 ����� � 1 �����. (� ������ ������ �� ����� ��� 1 ��� � 15 �����). � ������� �� ��������, ��� ��� ��������� ��� � 10 ���.
extern int MagicNumber = 777;
extern double MaxSpread = 4;
extern double OpenSlippage = 3; //--- ��������������� ��� �������� ������
extern double CloseSlippage = 5; //--- ��������������� ��� �������� ������
extern int RequoteAttempts = 5; //--- ������������ ����� ���������� ��� ��������/�������� ������ ��� �������� � ������ �������
extern bool WriteLog = TRUE; //--- //--- ��������� ����������� ���� � ���������.
extern bool WriteDebugLog = TRUE; //--- ��������� ����������� ���� �� ������� � ���������.
extern bool PrintLogOnChart = TRUE; //--- ��������� ������������ �� ������� (��� ������������ ����������� �������������)

//+--------------------------------------------------------------------------------------------------------------+
//| ���� �������������� ����������
//+--------------------------------------------------------------------------------------------------------------+

double pp;
int pd;
double cf;
string EASymbol; //--- ������� ������
int SP;
int CloseSP;
int MaximumTrades = 1;
double NDMaxSpread; //--- ������������ ����� ����� �������
bool CheckSpreadRuleBuy; //--- ������� ��� �������� ������ ����� ��������� (������������� ������������ ��������� � ����������� ������)
bool CheckSpreadRuleSell;
string OpenOrderComment = "WallstreetFX-theclai";
int RandomOpenTimePercent = 0; //--- ������������ ��� ������� ������ ������� ���������, ����������� ��������� �����. ���������� � ��������.
//---

//--- ��������� ��� ��������
double MinLot = 0.01;
double MaxLot = 0.01;
double LotStep = 0.01;
int LotValue = 100000;
double FreeMargin = 1000.0;
double LotPrice = 1;
double LotSize;

//--- ��������� �� ��������

int iWPR_Filter_OpenLong_a;
int iWPR_Filter_OpenLong_b;
int iWPR_Filter_OpenShort_a;
int iWPR_Filter_OpenShort_b;

//--- ��������� �� ��������

int iWPR_Filter_CloseLong;
int iWPR_Filter_CloseShort;

//---
color OpenBuyColor = Blue;
color OpenSellColor = Red;
color CloseBuyColor = DodgerBlue;
color CloseSellColor = DeepPink;


//+--------------------------------------------------------------------------------------------------------------+
//| INIT. ������������� ��������� ����������, �������� �������� �� �������.
//+--------------------------------------------------------------------------------------------------------------+
void init() {
//+--------------------------------------------------------------------------------------------------------------+

   //---
   if (IsTesting() && !IsVisualMode()) {PrintLogOnChart = FALSE; OneOrderInBarMode = FALSE;} //--- ���� ���������, �� ����������� ����������� �� ������� � ������� OneOrderInBarMode
   if (!PrintLogOnChart) Comment("");
   //---
   EASymbol = Symbol(); //--- ������������� �������� �������
   //---
   if (Digits < 4) {
      pp = 0.01;
      pd = 2;
      cf = 0.01;
   } else {
      pp = 0.0001;
      pd = 4;
      cf = 0.0001;
   }
   //---
   SP = OpenSlippage * MathPow(10, Digits - pd); //--- ������ ��������������� ���� ��� ���������
   CloseSP = CloseSlippage * MathPow(10, Digits - pd);
   NDMaxSpread = NormalizeDouble(MaxSpread * pp, pd + 1); //--- �������������� �������� MaxSpread � ������
   //---
   if (ObjectFind("BKGR") >= 0) ObjectDelete("BKGR");
   if (ObjectFind("BKGR2") >= 0) ObjectDelete("BKGR2");
   if (ObjectFind("BKGR3") >= 0) ObjectDelete("BKGR3");
   if (ObjectFind("BKGR4") >= 0) ObjectDelete("BKGR4");
   if (ObjectFind("LV") >= 0) ObjectDelete("LV");
   //---
   
   //--- ������������� ���������� ��� MM
   
   MinLot = MarketInfo(Symbol(), MODE_MINLOT);
   MaxLot = MarketInfo(Symbol(), MODE_MAXLOT);
   LotValue = MarketInfo(Symbol(), MODE_LOTSIZE);
   LotStep = MarketInfo(Symbol(), MODE_LOTSTEP);
   FreeMargin = MarketInfo(Symbol(), MODE_MARGINREQUIRED);
   
   //--- ��������� �������� ��������� ���� ����������� ������� ������ �� ���������� ������ �������.
   double SymbolBid = 0;
   if (StringSubstr(AccountCurrency(), 0, 3) == "JPY") {
      SymbolBid = MarketInfo("USDJPY" + StringSubstr(Symbol(), 6), MODE_BID);
      if (SymbolBid > 0.1) LotPrice = SymbolBid;
      else LotPrice = 84;
   }
   //---
   if (StringSubstr(AccountCurrency(), 0, 3) == "GBP") {
      SymbolBid = MarketInfo("GBPUSD" + StringSubstr(Symbol(), 6), MODE_BID);
      if (SymbolBid > 0.1) LotPrice = 1 / SymbolBid;
      else LotPrice = 0.6211180124;
   }
   //---
   if (StringSubstr(AccountCurrency(), 0, 3) == "EUR") {
      SymbolBid = MarketInfo("EURUSD" + StringSubstr(Symbol(), 6), MODE_BID);
      if (SymbolBid > 0.1) LotPrice = 1 / SymbolBid;
      else LotPrice = 0.7042253521;
   }
   
   //--- ��������� �� ��������
   
   iWPR_Filter_OpenLong_a = iWPR_Filter_Open_a;
   iWPR_Filter_OpenLong_b = iWPR_Filter_Open_b;
   iWPR_Filter_OpenShort_a = -100 - iWPR_Filter_Open_a;
   iWPR_Filter_OpenShort_b = -100 - iWPR_Filter_Open_b;

   //--- ��������� �� ��������
   
   iWPR_Filter_CloseLong = iWPR_Filter_Close;
   iWPR_Filter_CloseShort = -100 - iWPR_Filter_Close;
   
   //---
   return (0);
   
}

//+--------------------------------------------------------------------------------------------------------------+
//| DEINIT. �������� �������� �� �������.
//+--------------------------------------------------------------------------------------------------------------+
int deinit() {
//+--------------------------------------------------------------------------------------------------------------+

   if (ObjectFind("BKGR") >= 0) ObjectDelete("BKGR");
   if (ObjectFind("BKGR2") >= 0) ObjectDelete("BKGR2");
   if (ObjectFind("BKGR3") >= 0) ObjectDelete("BKGR3");
   if (ObjectFind("BKGR4") >= 0) ObjectDelete("BKGR4");
   if (ObjectFind("LV") >= 0) ObjectDelete("LV");
   //---
   return (0);
   
}

//+--------------------------------------------------------------------------------------------------------------+
//| START. �������� �� ������, � ����� ����� ������� Scalper.
//+--------------------------------------------------------------------------------------------------------------+
int start() {
//+--------------------------------------------------------------------------------------------------------------+
   
   if (PrintLogOnChart) ShowComments (); //--- ��������� ������������ �� �������
   //---
   CloseOrders(); //--- ������������� �������
   ModifyOrders(); //--- ����� � ���������
   
   //--- ������������� ������ �����
   if (AutoMM > 0.0 && (!RecoveryMode)) LotSize = MathMax(MinLot, MathMin(MaxLot, MathCeil(MathMin(AutoMM_Max, AutoMM) / LotPrice / 100.0 * AccountFreeMargin() / LotStep / (LotValue / 100)) * LotStep));
   if (AutoMM > 0.0 && RecoveryMode) LotSize = CalcLots(); //--- ���� ������� RecoveryMode ���������� CalcLots
   if (AutoMM == 0.0) LotSize = FixedLot;
   
   //--- �������� ������� ������������ ������ ��� ���������� M15
   if(iBars(Symbol(), PERIOD_M15) < iMA_Period || iBars(Symbol(), PERIOD_M15) < iWPR_Period || iBars(Symbol(), PERIOD_M15) < iATR_Period || iBars(Symbol(), PERIOD_M15) < iCCI_Period)
   {
      Print("������������ ������������ ������ ��� ��������");
      return;
   }
   //---
   if (DayOfWeek() == 1 && iVolume(NULL, PERIOD_D1, 0) < 5.0) return (0);
   if (StringLen(EASymbol) < 6) return (0);   
   //---
   if ((!IsTesting()) && IsStopped()) return (0);
   if ((!IsTesting()) && !IsTradeAllowed()) return (0);
   if ((!IsTesting()) && IsTradeContextBusy()) return (0);
   //---
   HideTestIndicators(TRUE);
   //---
   Scalper();
   //---
   return (0);
}

//+--------------------------------------------------------------------------------------------------------------+
//| Scalper. �������� �������. ������� ������������ �������� ������, ����� �������� �������� �� ����.
//+--------------------------------------------------------------------------------------------------------------+
void Scalper() {
//+--------------------------------------------------------------------------------------------------------------+

   bool OpenBuyRule = TRUE;
   bool OpenSellRule = TRUE;
   
   //--- ������� ��� �������� ��������������� �������.
   if (No_Hedge_Trades == TRUE && CheckOpenTrade(OP_SELL)) OpenBuyRule = FALSE;
   if (No_Hedge_Trades == TRUE && CheckOpenTrade(OP_BUY)) OpenSellRule = FALSE;
      
   //--- �������� �� �������� �������� ������
   if (OpenLongSignal() && !CheckOpenTrade(OP_BUY) && OpenBuyRule && OneOrderInBar(OP_BUY) && LongTrade) {
         
      //--- ��������� � ����������� ������
      if (MaxSpreadFilter()) {
         if (!CheckSpreadRuleBuy && WriteDebugLog) {
         //---
         Print("�������� ������ �� ������� �������� ��-�� �������� ������.");
         Print("������� ����� = ", DoubleToStr((Ask - Bid) / pp, 1), ",  MaxSpread = ", DoubleToStr(MaxSpread, 1));
         Print("������� WSFR 3.8.5 ����� ��������� �����, ����� ����� ������ ����������.");
         }
         //---
         CheckSpreadRuleBuy = TRUE;
      //---
      } else {
         //---
         CheckSpreadRuleBuy = FALSE;
         // ������ �������� ��� ���������� �������������
         if (CheckVolatility()) 
         {
            Print("������������� ���������, ��� ������������ ���������� �������� ������");
         }else
         { 
           OpenPosition(OP_BUY);
         }
      }
   }//--- �������� if (OpenLongSignal()
      
   //--- �������� �� �������� ��������� ������   
   if (OpenShortSignal()&& !CheckOpenTrade(OP_SELL) && OpenSellRule && OneOrderInBar(OP_SELL) && ShortTrade) {
         
      //--- ��������� � ����������� ������
      if (MaxSpreadFilter()) {
         if (!CheckSpreadRuleSell && WriteDebugLog) {
         //---
         Print("�������� ������ �� ������� �������� ��-�� �������� ������.");
         Print("������� ����� = ", DoubleToStr((Ask - Bid) / pp, 1), ",  MaxSpread = ", DoubleToStr(MaxSpread, 1));
         Print("������� WSFR 3.8.5 ����� ��������� �����, ����� ����� ������ ����������.");
         }
         //---
         CheckSpreadRuleSell = TRUE;
      //---
      } else {
         //---
         CheckSpreadRuleSell = FALSE;
         // ������ �������� ��� ���������� �������������
         if (CheckVolatility()) 
         {
            Print("������������� ���������, ��� ������������ ���������� �������� ������");
         }else
         { 
           OpenPosition(OP_SELL);
         }
      }
   } //--- ��������  if (OpenShortSignal()
}

//+--------------------------------------------------------------------------------------------------------------+
//| ������ �������������
//+--------------------------------------------------------------------------------------------------------------+
bool CheckVolatility() {
   double HeightFilter_a = NormalizeDouble(VolaFilter * pp, pd);
   bool restrict = false;
   if (NormalizeDouble(iHigh(NULL, PERIOD_M15, 1) - iLow(NULL, PERIOD_M15, 1), pd) > HeightFilter_a) restrict = true;
   if (NormalizeDouble(iHigh(NULL, PERIOD_M15, 2) - iLow(NULL, PERIOD_M15, 2), pd) > HeightFilter_a) restrict = true;
   return (restrict);
}

//+--------------------------------------------------------------------------------------------------------------+
//| OpenPosition. ������� �������� �������.
//+--------------------------------------------------------------------------------------------------------------+
int OpenPosition(int OpType) {
//+--------------------------------------------------------------------------------------------------------------+

   int RandomOpenTime; //--- �������� �� ������� �� ��������
   color OpenColor; //--- ���� �������� �������. ���� Buy �� �������, ���� Sell �� �������
   int OpenOrder = 0; //--- �������� �������
   int OpenOrderError; //--- ������ ��������
   string OpTypeString; //--- �������������� ���� ������� � ��������� ��������
   double OpenPrice; //--- ���� ��������
   int    maxtry = RequoteAttempts;
   int    lasterror = 0;
   double price = 0;
   //---
   double TP, SL;
   double OrderTP = NormalizeDouble (TakeProfit * pp , pd); //--- ����������� ����-������ � ��� Points
   double OrderSL = NormalizeDouble (StopLoss * pp , pd); //--- ����������� ����-���� � ��� Points
     
   //--- �������� � �������� ����� ����������
   if (RandomOpenTimePercent > 0) {
      MathSrand(TimeLocal());
      RandomOpenTime = MathRand() % RandomOpenTimePercent;
      
      if (WriteLog) {
      Print("DelayRandomiser: �������� ", RandomOpenTime, " ������.");
      }
      
      Sleep(1000 * RandomOpenTime);
   } //--- �������� if (RandomOpenTimePerc
   
   double OpenLotSize = LotSize; //--- ������ ������ �������
   
   //--- ���� �� ������ �������, ���������� ������
   if (AccountFreeMarginCheck(EASymbol, OpType, OpenLotSize) <= 0.0 || GetLastError() == 134/* NOT_ENOUGH_MONEY */) {
      //---
      if (WriteDebugLog) {
      //---
         Print("��� �������� ������ ������������ ��������� �����.");
         Comment("��� �������� ������ ������������ ��������� �����.");
      //---
      }
      return (-1);
   } //--- �������� if (AccountFreeMarginCheck  
   
   RefreshRates();
   
   //--- ���� ������� �������, ��
   if (OpType == OP_BUY) {
      OpenPrice = NormalizeDouble(Ask, Digits);
      OpenColor = OpenBuyColor;
      
      //---
      if (UseStopLevels) { //--- ���� �������� ����-������ (����-���� � ����-������)
      
      TP = NormalizeDouble(OpenPrice + OrderTP, Digits); //--- �� ����������� ����-������
      SL = NormalizeDouble(OpenPrice - OrderSL, Digits); //--- � ����-����
      //---
      } else {TP = 0; SL = 0;}
   
   //--- ���� �������� �������, ��   
   } else {
      OpenPrice = NormalizeDouble(Bid, Digits);
      OpenColor = OpenSellColor;
      
      //---
      if (UseStopLevels) {
       
      TP = NormalizeDouble(OpenPrice - OrderTP, Digits);
      SL = NormalizeDouble(OpenPrice + OrderSL, Digits);
      }
      //---
      else {TP = 0; SL = 0;}
   }
   
//--- ���� ��� ���������� Market Execution (�������� ����������), �� ������� ��������� ����� ��� sl � tp, � ����� ������������ ���

if (IsMarketExecution && UseStopLevels)
   {
   OpenOrder = OrderSend(EASymbol, OpType, OpenLotSize, OpenPrice, SP, 0, 0, OpenOrderComment, MagicNumber, 0, OpenColor);
   if (OpenOrder > 0)
      {
      OrderModify(OpenOrder,OrderOpenPrice(),SL,TP,0);
      return(OpenOrder);
      }
   }
   
      //--- ���� �� ���, �� ����� ��������� � sl � tp
      
      else
   {
   OpenOrder = OrderSend(EASymbol, OpType, OpenLotSize, OpenPrice, SP, SL, TP, OpenOrderComment, MagicNumber, 0, OpenColor);
   if (OpenOrder > 0) return(OpenOrder);
   }

//--- ���� ��� ������ �� �����, �� ��������� ��������.

if ((OpType != OP_BUY) && (OpType != OP_SELL)) return(OpenOrder);

//--- ���� ����� �������� �������, �� ���������� ��������� �� e-mail (���� �������� ��������)

if (OpenOrder < 0) { //--- ���� ����� �� ��������, ��
   OpenOrderError = GetLastError(); //--- ���������� ������
         //---
   if (WriteDebugLog) {
      if (OpType == OP_BUY) OpTypeString = "OP_BUY";
         else OpTypeString = "OP_SELL";
            Print("��������: OrderSend(", OpTypeString, ") ������ = ", ErrorDescription(OpenOrderError)); //--- ��� ������ �� �������
         } //--- �������� if (WriteDebugLog)
}

//--- ��� �������� ��������� ��������.

lasterror = GetLastError();

if ((OpenOrder < 0) && ((lasterror == 135) || (lasterror == 138) || (lasterror == 146)))
   {
   Print("Requote. Error" + lasterror + ". Ticket: " + OpenOrder);
   }
      else
   {
   return(OpenOrder);
   }

//--- ���� �������� ������ ��� ������������� ������ (������������ ����� ������� �������� ��� ������������� ������ ����� �������� RequoteAttempts) 

price = OpenPrice;

for (int attempt = 1; attempt <= maxtry; attempt++)
   {
   RefreshRates();
   if (OpType == OP_BUY)
      {
      if (Ask <= price)
         {
         if (IsMarketExecution && UseStopLevels)
            {
            OpenOrder = OrderSend(EASymbol, OpType, OpenLotSize, NormalizeDouble(Ask,Digits), SP, 0, 0, OpenOrderComment, MagicNumber, 0, OpenColor);
            if (OpenOrder > 0)
               {
               OrderModify(OpenOrder,OrderOpenPrice(),SL,TP,0);
               return(OpenOrder);
               }
            }
               else
            {
            OpenOrder = OrderSend(EASymbol, OpType, OpenLotSize, NormalizeDouble(Ask,Digits), SP, SL, TP, OpenOrderComment, MagicNumber, 0, OpenColor);
            if (OpenOrder > 0) return(OpenOrder);
            }
         if ((GetLastError() != 135) && (GetLastError() != 138) && (GetLastError() != 146)) return(OpenOrder);
         Print("Requote. " + "Attempt " + (attempt + 1));
         continue;
         }
      }
   if (OpType == OP_SELL)
      {
      if (Bid >= price)
         {
         if (IsMarketExecution && UseStopLevels)
            {
            OpenOrder = OrderSend(EASymbol, OpType, OpenLotSize, NormalizeDouble(Bid,Digits), SP, 0, 0, OpenOrderComment, MagicNumber, 0, OpenColor);
            if (OpenOrder > 0)
               {
               OrderModify(OpenOrder,OrderOpenPrice(),SL,TP,0);
               return(OpenOrder);
               }
            }
               else
            {
            OpenOrder = OrderSend(EASymbol, OpType, OpenLotSize, NormalizeDouble(Bid,Digits), SP, SL, TP, OpenOrderComment, MagicNumber, 0, OpenColor);
            if (OpenOrder > 0) return(OpenOrder);
            }
         if ((GetLastError() != 135) && (GetLastError() != 138) && (GetLastError() != 146)) return(OpenOrder);
         Print("Requote. " + "Attempt " + (OpenOrder + 1));
         }
      }
   }

//---
return(-1);

}

//+--------------------------------------------------------------------------------------------------------------+
//| ModifyOrders. ����������� ������� � ���������.
//+--------------------------------------------------------------------------------------------------------------+
void ModifyOrders() {
//+--------------------------------------------------------------------------------------------------------------+

   int total = OrdersTotal() - 1;
   int ModifyError = GetLastError();
   
   //---
   for (int i = total; i >= 0; i--) { //--- ������� �������� �������
      if (!OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) {
         if (WriteDebugLog) Print("��������� ������ �� ����� ������� ������. �������: ", ErrorDescription(ModifyError));
      
      } else {
      
      //--- ����������� ������ �� �������
      if (OrderType() == OP_BUY) {
         if (OrderMagicNumber() == MagicNumber && OrderSymbol() == EASymbol) {
            if (Bid - OrderOpenPrice() > SecureProfitTriger * pp && MathAbs(OrderOpenPrice() + SecureProfit * pp - OrderStopLoss()) >= Point) {
               //--- ������������ �����
               ModifyOrder(EASymbol, OrderOpenPrice(), NormalizeDouble(OrderOpenPrice() + SecureProfit * pp, Digits), OrderTakeProfit(), Blue);
               }
            }
         } //--- �������� if (OrderType() == OP_BUY)
      
      //--- ����������� ������ �� �������
      if (OrderType() == OP_SELL) {
         if (OrderMagicNumber() == MagicNumber && OrderSymbol() == EASymbol) {
            if (OrderOpenPrice() - Ask > SecureProfitTriger * pp && MathAbs(OrderOpenPrice() - SecureProfit * pp - OrderStopLoss()) >= Point) {
               //--- ������������ �����
               ModifyOrder(EASymbol, OrderOpenPrice(), NormalizeDouble(OrderOpenPrice() - SecureProfit * pp, Digits), OrderTakeProfit(), Red);
               }
            }
         } //--- �������� if (OrderType() == OP_SELL)
      }
   } //--- �������� for (int i = total - 1; i >= 0; i--)
}

//+--------------------------------------------------------------------------------------------------------------+
//| ModifyOrder. ����������� �������������� ���������� ������.
//|  
//| ���������:
//|   sy - ������������ �����������  ("" - ������� ������)
//|   pp - ���� �������� �������, ��������� ������
//|   sl - ������� ������� �����
//|   tp - ������� ������� �����
//|   cl - ����
//+--------------------------------------------------------------------------------------------------------------+
void ModifyOrder(string sy="", double pp=-1, double sl=0, double tp=0, color cl=CLR_NONE) {
//+--------------------------------------------------------------------------------------------------------------+

   int ModifyTicketID = OrderTicket();
   
   if (sy=="") sy=Symbol();
   bool   fm; //--- ����������� ������
   double pAsk=MarketInfo(sy, MODE_ASK);
   double pBid=MarketInfo(sy, MODE_BID);
   int    dg, err, it;
   int    PauseAfterError = 5; //--- ����� � �������� ����� ��������� �����������
   int    NumberOfTry = 10; //--- ���-�� ������� ����������� ��� ������������� ������
   
   //--- �������� �� ������ ���������� 
   if (pp<=0) pp=OrderOpenPrice();
   if (sl<0) sl=OrderStopLoss();
   if (tp<0) tp=OrderTakeProfit();
   
   //--- ������������� ���������� 
   dg=MarketInfo(sy, MODE_DIGITS);
   pp=NormalizeDouble(pp, dg);
   sl=NormalizeDouble(sl, dg);
   tp=NormalizeDouble(tp, dg);
   
   //--- �������������� �������� �� ������ ����������, ����� �������
   if (pp!=OrderOpenPrice() || sl!=OrderStopLoss() || tp!=OrderTakeProfit()) {
      
      //--- �������� ���� ������� �����������
      for (it=1; it<=NumberOfTry; it++) {
         if (!IsTesting() && (!IsExpertEnabled() || IsStopped())) break;
         while (!IsTradeAllowed()) Sleep(5000);
         RefreshRates();
         
         //--- ������������ �����
         fm=OrderModify(OrderTicket(), pp, sl, tp, 0, cl);
         
         //--- ���� ��������� ������, ��
         if (!fm) {
         err=GetLastError();
         
         //--- ����� ������, ���� �������� ����������� ������
         if (WriteDebugLog) Print("��������� ������ �� ����� ����������� ������ (", GetNameOP(OrderType()), ",", ModifyTicketID, "). �������: ", ErrorDescription(err),". ������� �",it);
         
         //--- ��� PauseAfterError ������, ����� ���� ��������� ������� �����������
         Sleep(1000*PauseAfterError);
         
         } //--- �������� if (!fm) {
      }
   }
}

//+--------------------------------------------------------------------------------------------------------------+
//| CloseTrades. ����������� ����-������ � ����-����.
//| ���� �������� ������� UseStopLevels, �� ������������ ��� ������� ���������� ��������.
//+--------------------------------------------------------------------------------------------------------------+
void CloseOrders() {
//+--------------------------------------------------------------------------------------------------------------+

   int total = OrdersTotal() - 1;
   int SelectCloseError = GetLastError();
   
   //---
   for (int i = total; i >= 0; i--) { //--- ������� �������� �������
      if (!OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) {
         if (WriteDebugLog) Print("��������� ������ �� ����� ������� ������. �������: ", ErrorDescription(SelectCloseError));
      
      } else {
      
      //--- �������� �� ������� ��� ����� ������� �� �������
      if (OrderType() == OP_BUY) {
         if (OrderMagicNumber() == MagicNumber && OrderSymbol() == EASymbol) {
            if (Bid >= OrderOpenPrice() + TakeProfit * pp || Bid <= OrderOpenPrice() - StopLoss * pp || CloseLongSignal(OrderOpenPrice(), ExistPosition())) {
               //---
               CloseOrder(OrderTicket(),Bid);
               //---
            }
         }
      } //--- �������� if (OrderType() == OP_BUY)
      
      //--- �������� �� ������� ��� ����� ������� �� �������
      if (OrderType() == OP_SELL) {
         if (OrderMagicNumber() == MagicNumber && OrderSymbol() == EASymbol) {
            if (Ask <= OrderOpenPrice() - TakeProfit * pp || Ask >= OrderOpenPrice() + StopLoss * pp || CloseShortSignal(OrderOpenPrice(), ExistPosition())) {
               //---
               CloseOrder(OrderTicket(),Ask);
               //---
               }
            }
         } //--- �������� if (OrderType() == OP_SELL)
      } 
   } //--- �������� for (int i = total - 1; i >= 0; i--) {
}

//+--------------------------------------------------------------------------------------------------------------+
//| CloseOrder. ������� �������� ������.
//+--------------------------------------------------------------------------------------------------------------+
int CloseOrder(int ticket, double prce) {
//+--------------------------------------------------------------------------------------------------------------+

//--- ������������� ���������� ����������� ��� ������� �������� ��� �������� ��� ������� �������.

double price;
int    slippage;
double p = prce;
int    maxtry = RequoteAttempts;
color  CloseColor;

OrderSelect(ticket,SELECT_BY_TICKET,MODE_TRADES);

int ordtype = OrderType();
if (ordtype == OP_BUY) {price = NormalizeDouble(Bid,Digits); CloseColor = CloseBuyColor;}
if (ordtype == OP_SELL) {price = NormalizeDouble(Ask,Digits); CloseColor = CloseSellColor;}

if (MathAbs(OrderTakeProfit() - price) <= MarketInfo(Symbol(),MODE_FREEZELEVEL) * Point) return(0); //--- �������� ������� ��������� ����-�������
if (MathAbs(OrderStopLoss() - price) <= MarketInfo(Symbol(),MODE_FREEZELEVEL) * Point) return(0); //--- �������� ������� ��������� ����-�����

if (OrderClose(ticket,OrderLots(),price,CloseSlippage,CloseColor)) return(1); //--- ���� ����� ������ �������, �� ���������� 1)
if ((GetLastError() != 135) && (GetLastError() != 138) && (GetLastError() != 146)) return(0); //--- ���� ��� �� 0

Print("Requote");

//--- ���� �������� ������ ��� ������������� ������ (������������ ����� ������� �������� ��� ������������� ������ ����� �������� RequoteAttempts) 

for (int attempt = 1; attempt <= maxtry; attempt++)
   {
   RefreshRates();
   if (ordtype == OP_BUY)
      {
      slippage = MathRound((Bid - p) / pp);
      if (Bid >= p)
         {
         Print("Closing order. Attempt " + (attempt + 1));
         if (OrderClose(ticket,OrderLots(),NormalizeDouble(Bid,Digits),slippage,CloseColor)) return(1);
         if (!((GetLastError() != 135) && (GetLastError() != 138) && (GetLastError() != 146))) continue;
         return(0);
         }
      }
   if (ordtype == OP_SELL)
      {
      slippage = MathRound((p - Ask) / pp);
      if (p >= Ask)
         {
         Print("Closing order. Attempt " + (attempt + 1));
         if (OrderClose(ticket,OrderLots(),NormalizeDouble(Ask,Digits),slippage,CloseColor)) return(1);
         if ((GetLastError() != 135) && (GetLastError() != 138) && (GetLastError() != 146)) return(0);
         }
      }
   }
}

//+--------------------------------------------------------------------------------------------------------------+
//| OpenLongSignal. ������ �� �������� ������� �������.
//+--------------------------------------------------------------------------------------------------------------+
bool OpenLongSignal() {
//+--------------------------------------------------------------------------------------------------------------+

bool result = false;
bool result1 = false;
bool result2 = false;
bool result3 = false;

//--- ������ �������� �������� �� ����
double iClose_Signal = iClose(NULL, PERIOD_M15, 1);
double iMA_Signal = iMA(NULL, PERIOD_M15, iMA_Period, 0, MODE_SMMA, PRICE_CLOSE, 1);
double iWPR_Signal = iWPR(NULL, PERIOD_M15, iWPR_Period, 1);
double iATR_Signal = iATR(NULL, PERIOD_M15, iATR_Period, 1);
double iCCI_Signal = iCCI(NULL, PERIOD_M15, iCCI_Period, PRICE_TYPICAL, 1);
//---
double iMA_Filter_a = NormalizeDouble(iMA_Filter_Open_a*pp,pd);
double iMA_Filter_b = NormalizeDouble(iMA_Filter_Open_b*pp,pd);
double BidPrice = Bid; //--- (iClose_Signal >= BidPrice) ��������� ��� ������ � Bid (� �� � Ask, ��� ������ ����), ��� ��� ���� �������� ����� iClose_Signal ����������� �� ��������� �������� Bid
//---

//--- ������� ������ �� ��� � ��� ��������
if (iATR_Signal <= FilterATR * pp) return (0);
//---
if (iClose_Signal - iMA_Signal > iMA_Filter_a && iClose_Signal - BidPrice >= - cf && iWPR_Filter_OpenLong_a > iWPR_Signal) result1 = true;
else result1 = false;
//---
if (iClose_Signal - iMA_Signal > iMA_Filter_b && iClose_Signal - BidPrice >= - cf && - iCCI_OpenFilter > iCCI_Signal) result2 = true;
else result2 = false;
//---
if (iClose_Signal - iMA_Signal > iMA_Filter_b && iClose_Signal - BidPrice >= - cf && iWPR_Filter_OpenLong_b > iWPR_Signal) result3 = true;
else result3 = false;
//---
if (result1 == true || result2 == true || result3 == true) result = true;
else result = false;
//---
return (result);

}

//+--------------------------------------------------------------------------------------------------------------+
//| OpenShortSignal. ������ �� �������� �������� �������.
//+--------------------------------------------------------------------------------------------------------------+
bool OpenShortSignal() {
//+--------------------------------------------------------------------------------------------------------------+

bool result = false;
bool result1 = false;
bool result2 = false;
bool result3 = false;

//--- ������ �������� �������� �� ����
double iClose_Signal = iClose(NULL, PERIOD_M15, 1);
double iMA_Signal = iMA(NULL, PERIOD_M15, iMA_Period, 0, MODE_SMMA, PRICE_CLOSE, 1);
double iWPR_Signal = iWPR(NULL, PERIOD_M15, iWPR_Period, 1);
double iATR_Signal = iATR(NULL, PERIOD_M15, iATR_Period, 1);
double iCCI_Signal = iCCI(NULL, PERIOD_M15, iCCI_Period, PRICE_TYPICAL, 1);
//---
double iMA_Filter_a = NormalizeDouble(iMA_Filter_Open_a*pp,pd);
double iMA_Filter_b = NormalizeDouble(iMA_Filter_Open_b*pp,pd);
double BidPrice = Bid;
//---

//--- ������� ������ �� ��� � ��� ��������
if (iATR_Signal <= FilterATR * pp) return (0);
//---
if (iMA_Signal - iClose_Signal > iMA_Filter_a && iClose_Signal - BidPrice <= cf && iWPR_Signal > iWPR_Filter_OpenShort_a) result1 = true;
else result1 = false;
//---
if (iMA_Signal - iClose_Signal > iMA_Filter_b && iClose_Signal - BidPrice <= cf && iCCI_Signal > iCCI_OpenFilter) result2 = true;
else result2 = false;
//---
if (iMA_Signal - iClose_Signal > iMA_Filter_b && iClose_Signal - BidPrice <= cf && iWPR_Signal > iWPR_Filter_OpenShort_b) result3 = true;
else result3 = false;
//---
if (result1 == true || result2 == true || result3 == true) result = true;
else result = false;
//---
return (result);

}

//+--------------------------------------------------------------------------------------------------------------+
//| CloseLongSignal. ������ �� �������� ������� �������.
//+--------------------------------------------------------------------------------------------------------------+
bool CloseLongSignal(double OrderPrice, int CheckOrders) {
//+--------------------------------------------------------------------------------------------------------------+

bool result = false;
bool result1 = false;
bool result2 = false;
//---
double iWPR_Signal = iWPR(NULL, PERIOD_M15, iWPR_Period, 1);
double iClose_Signal = iClose(NULL, PERIOD_M15, 1);
double iOpen_CloseSignal = iOpen(NULL, PERIOD_M1, 1);
double iClose_CloseSignal = iClose(NULL, PERIOD_M1, 1);
//---
double MaxLoss = NormalizeDouble(-MaxLossPoints * pp,pd);
//---
double Price_Filter = NormalizeDouble(Price_Filter_Close*pp,pd);
double BidPrice = Bid;
//---

//---
if (OrderPrice - BidPrice <= MaxLoss && iClose_Signal - BidPrice <= cf && iWPR_Signal > iWPR_Filter_CloseLong && CheckOrders == 1) result1 = true;
else result1 = false;
//---
if (iOpen_CloseSignal > iClose_CloseSignal && BidPrice - OrderPrice >= Price_Filter && CheckOrders == 1) result2 = true;
else result2 = false;
//---
if (result1 == true || result2 == true) result = true;
else result = false;
//---
return (result);

}

//+--------------------------------------------------------------------------------------------------------------+
//| CloseShortSignal. ������ �� �������� �������� �������.
//+--------------------------------------------------------------------------------------------------------------+
bool CloseShortSignal(double OrderPrice, int CheckOrders) {
//+--------------------------------------------------------------------------------------------------------------+

bool result = false;
bool result1 = false;
bool result2 = false;
//---
double iWPR_Signal = iWPR(NULL, PERIOD_M15, iWPR_Period, 1);
double iClose_Signal = iClose(NULL, PERIOD_M15, 1);
double iOpen_CloseSignal = iOpen(NULL, PERIOD_M1, 1);
double iClose_CloseSignal = iClose(NULL, PERIOD_M1, 1);
//---
double MaxLoss = NormalizeDouble(-MaxLossPoints*pp,pd);
//---
double Price_Filter = NormalizeDouble(Price_Filter_Close*pp,pd);
double BidPrice = Bid;
double AskPrice = Ask;
//---

//---
if (AskPrice - OrderPrice <= MaxLoss && iClose_Signal - BidPrice >= - cf && iWPR_Signal < iWPR_Filter_CloseShort && CheckOrders == 1) result1 = true;
else result1 = false;
//---
if (iOpen_CloseSignal < iClose_CloseSignal && OrderPrice - AskPrice >= Price_Filter && CheckOrders == 1) result2 = true;
else result2 = false;
//---
if (result1 == true || result2 == true) result = true;
else result = false;
//---
return (result);

}

//+--------------------------------------------------------------------------------------------------------------+
//| CalcLots. ������� ������� ������ ����.
//| ��� AutoMM > 0.0 && RecoveryMode ������� CalcLots ����������� ����� ���� ������������ ��������� �������.
//| 
//| ����� ������ ���� ������������� ������ �� ����� �������� � ������� �������. �� ���� ���������� ���� ������
//| ������� �� ������ �� ��������� �������, �� � �� ����� �������� � ������� ���������� �������.
//| 
//| ������ �������� ��, ������� ������������ ��� ������ �� ������������ ����� ����-������ ��� ����������
//| ��������� RecoveryMode, �� ����, ��� ������� ����� �������� ����� �������������� ��������.
//+--------------------------------------------------------------------------------------------------------------+
double CalcLots() {
//+--------------------------------------------------------------------------------------------------------------+

   double SumProfit; //--- ��������� ������
   int OldOrdersCount; //--- ������� ���-�� �������� ���������� �������
   double loss; //--- ��������
   int LossOrdersCount; //--- ����� ����� � �������
   double pr; //--- ������
   int ProfitOrdersCount; //--- ���-�� ���������� ������� � �������
   double LastPr; //--- ���������� �������� ������
   int LastCount; //--- ���������� �������� �������� �������
   double MultiLot = 1; //---  ��������� �������� ��������� ����
   //---
   
   //--- ���� �� �������, ��
   if (MultiLotPercent > 0.0 && AutoMM > 0.0) {
      
      //--- �������� ��������
      SumProfit = 0;
      OldOrdersCount = 0;
      loss = 0;
      LossOrdersCount = 0;
      pr = 0;
      ProfitOrdersCount = 0;
      //---
      
      //--- �������� �������� ����� ������
      for (int i = OrdersHistoryTotal() - 1; i >= 0; i--) {
         if (OrderSelect(i, SELECT_BY_POS, MODE_HISTORY)) {
            if (OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber) {
               OldOrdersCount++; //--- ������� ������
               SumProfit += OrderProfit(); //--- � ��������� ������
               
               //--- ���� ��������� ������ ������ pr (��� ������ ������ 0)
               if (SumProfit > pr) {
                  //--- �������������� ������ � ������� ���������� �������
                  pr = SumProfit;
                  ProfitOrdersCount = OldOrdersCount;
               }
               //--- ���� ��������� ������ ������ loss (��� ������ ������ 0)
               if (SumProfit < loss) {
                  //--- �������������� �������� � ������� ��������� �������
                  loss = SumProfit;
                  LossOrdersCount = OldOrdersCount;
               }
               //--- ���� ������� ���-�� ������������ ������� ������ ��� ����� MaxAnalizCount (50), �� � ������� ������� ������ ���������� ������ � ������ ��������.
               if (OldOrdersCount >= MaxAnalizCount) break;
            }
         }
      } //--- �������� for (int i = OrdersHistoryTotal() - 1; i >= 0; i--) {
      
      
      //--- ���� ����� ���������� ������� ������ ��� ����� ����� �����, �� ����������� �������� ��������� ���� MultiLot
      if (ProfitOrdersCount <= LossOrdersCount) MultiLot = MathPow(MultiLotPercent, LossOrdersCount);
      
      //--- ���� ���, ��
      else {
         
         //--- �������������� ��������� �� �������
         SumProfit = pr;
         OldOrdersCount = ProfitOrdersCount;
         LastPr = pr;
         LastCount = ProfitOrdersCount;
         
         //--- �������� �������� ����� ������ (����� ����� ���������� �������)
         for (i = OrdersHistoryTotal() - ProfitOrdersCount - 1; i >= 0; i--) {
            if (OrderSelect(i, SELECT_BY_POS, MODE_HISTORY)) {
               if (OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber) {
                  //--- ���� ������� ����� 50 ������� ����������� ��������
                  if (OldOrdersCount >= MaxAnalizCount) break;
                  //---
                  OldOrdersCount++; //--- ������� ���-�� �������
                  SumProfit += OrderProfit(); //--- � ������
                  
                  //--- ���� ����� ������ ������ ����������� (LastPr), ��
                  if (SumProfit < LastPr) {
                     //--- ������������������ �������� ������� � ���-�� �������
                     LastPr = SumProfit;
                     LastCount = OldOrdersCount;
                  }
               }
            }
         } //--- �������� for (i = OrdersHistoryTotal() - ProfitOrdersCount - 1; i >= 0; i--) {
         
         //--- ���� �������� �������� LastCount ����� �������� ���������� ������� ��� ������� ������ ����� �������, ��
         if (LastCount == ProfitOrdersCount || LastPr == pr) MultiLot = MathPow(MultiLotPercent, LossOrdersCount); //--- ����������� �������� ��������� ���� MultiLot
         
         //--- ���� ���, ��
         else {
            //--- ����� ������������� (loss - pr) �� ������������� (LastPr - pr) � ���������� � ������, ����� ����������� ��������� ���� MultiLot
            if (MathAbs(loss - pr) / MathAbs(LastPr - pr) >= (Risk + 100.0) / 100.0) MultiLot = MathPow(MultiLotPercent, LossOrdersCount);
            else MultiLot = MathPow(MultiLotPercent, LastCount);
         }
      }
   } //--- �������� if (MultiLotPercent > 0.0 && AutoMM > 0.0) {
   
   //--- �������� ��������� ����� ����, ������ �� ����������� ���� ��������
   for (double OpLot = MathMax(MinLot, MathMin(MaxLot, MathCeil(MathMin(AutoMM_Max, MultiLot * AutoMM) / 100.0 * AccountFreeMargin() / LotStep / (LotValue / 100)) * LotStep)); OpLot >= 2.0 * MinLot &&
      1.05 * (OpLot * FreeMargin) >= AccountFreeMargin(); OpLot -= MinLot) {
   }
   return (OpLot);
}

//+--------------------------------------------------------------------------------------------------------------+
//| MaxSpreadFilter. ������� ��� ������� ������� ������ � ��������� ��� �� ��������� MaxSpread.
//| ���� ������� ����� ��������, �� ���������� TRUE.
//+--------------------------------------------------------------------------------------------------------------+
bool MaxSpreadFilter() {
//+--------------------------------------------------------------------------------------------------------------+

   RefreshRates();
   if (NormalizeDouble(Ask - Bid, Digits) > NDMaxSpread) return (TRUE);
   //---
   else return (FALSE);
}

//+--------------------------------------------------------------------------------------------------------------+
//| ExistPosition. ������� �������� �������� �������.
//| ���� ������ ����� ���������� True, ���� ���, ���� ���������� (False, 0) �� ��������.
//+--------------------------------------------------------------------------------------------------------------+
int ExistPosition() {
//+--------------------------------------------------------------------------------------------------------------+

   int trade = OrdersTotal() - 1;
   for (int i = trade; i >= 0; i--) {
      if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) {
         if (OrderMagicNumber() == MagicNumber) {
            if (OrderSymbol() == EASymbol)
               if (OrderType() <= OP_SELL) return (1);
         }
      }
   }
   //---
   return (0);
}

//+--------------------------------------------------------------------------------------------------------------+
//| OneOrderInBar. ������� ���������, ���������� �� ����� ������ ������� �����.
//+--------------------------------------------------------------------------------------------------------------+
bool OneOrderInBar(int OpType = -1){
//+--------------------------------------------------------------------------------------------------------------+
   
   //--- ���� ��������� �������, �� ������ �� �����������.
   if (OneOrderInBarMode == FALSE) return(True);
   
   int Bar = Period(); //--- �����
   
   //--- ������ ������� �� �������� �������
   for(int i = OrdersHistoryTotal(); i>=0; i--){
      //---
      if(OrderSelect(i,SELECT_BY_POS,MODE_HISTORY)){
         //---
         if(OrderSymbol() == EASymbol && OrderType() == OpType && OrderMagicNumber() == MagicNumber) {
            
            //--- ���� ����� �������� ������ ������ ������� �������� �������� ����, �� ��������� �������� ������ ������.
            if(OrderCloseTime()>iTime(EASymbol,Bar,0)) return(False);
            }
         }
      }

   //---
   return(True);
}

//+--------------------------------------------------------------------------------------------------------------+
//| CheckOpenTrade. ������� ��� �������� ��������� ������. ��������� ��� �� ������ ����� �� OrderType.
//| ���� ��� ������, �� ���������� TRUE, ���� ���, �� FALSE.
//+--------------------------------------------------------------------------------------------------------------+
bool CheckOpenTrade(int OpType) {
//+--------------------------------------------------------------------------------------------------------------+
   
   int total = OrdersTotal();
   for (int i = total - 1; i >= 0; i--) {
      if (OrderSelect(i, SELECT_BY_POS) == TRUE)
         if (OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber && OrderType() == OpType) return (TRUE);
   }
   //---
   return (FALSE);
}

//+--------------------------------------------------------------------------------------------------------------+
//| ShowComments. ������� ��� ����������� ������������ �� �������.
//+--------------------------------------------------------------------------------------------------------------+
void ShowComments() {
//+--------------------------------------------------------------------------------------------------------------+

string ComSpacer = ""; //--- "/n"
datetime MyOpDate = TIME_DATE; //--- ����� � ����������� ���� ����������� (������)
//---
ComSpacer = ComSpacer
      + "\n  " 
      + "\n "
      + "\n  Version 3.8.5 (FINAL)"
      + "\n  Copyright � HELLTEAM^Pirat"
      + "\n  http://www.fxmania.ru"
      + "\n -----------------------------------------------"
      + "\n  Sets for: " + Symbol_Op
      + "\n  Optimization date: " + TimeToStr (Date, MyOpDate)
      + "\n -----------------------------------------------" 
      + "\n  SL = " + StopLoss + " pips / TP = " + TakeProfit + " pips" 
   + "\n  Spread = " + DoubleToStr((Ask - Bid) / pp, 1) + " pips";
   if (NormalizeDouble(Ask - Bid, Digits) > NDMaxSpread) ComSpacer = ComSpacer + " - TOO HIGH";
   else ComSpacer = ComSpacer + " - NORMAL";
   ComSpacer = ComSpacer 
   + "\n -----------------------------------------------";
   if (AutoMM > 0.0) {
      ComSpacer = ComSpacer 
         + "\n  AutoMM - ENABLED" 
      + "\n  Risk = " + DoubleToStr(AutoMM, 1) + "%";
   }
   ComSpacer = ComSpacer 
   + "\n  Trading Lots = " + DoubleToStr(LotSize, 2);
   ComSpacer = ComSpacer 
   + "\n -----------------------------------------------";
   if (UseStopLevels) {
      ComSpacer = ComSpacer 
      + "\n  Stop Levels - ENABLED";
   } else {
      ComSpacer = ComSpacer 
      + "\n  Stop Levels - DISABLED";
   }
      if (RecoveryMode) {
      ComSpacer = ComSpacer 
      + "\n  Recovery Mode - ENABLED";
   } else {
      ComSpacer = ComSpacer 
      + "\n  Recovery Mode - DISABLED";
   }
   ComSpacer = ComSpacer 
   + "\n -----------------------------------------------";
   Comment(ComSpacer);
   
   if (ObjectFind("LV") < 0) {
      ObjectCreate("LV", OBJ_LABEL, 0, 0, 0);
      ObjectSetText("LV", "WALL STREET ROBOT", 9, "Tahoma Bold", White);
      ObjectSet("LV", OBJPROP_CORNER, 0);
      ObjectSet("LV", OBJPROP_BACK, FALSE);
      ObjectSet("LV", OBJPROP_XDISTANCE, 13);
      ObjectSet("LV", OBJPROP_YDISTANCE, 23);
   }
   if (ObjectFind("BKGR") < 0) {
      ObjectCreate("BKGR", OBJ_LABEL, 0, 0, 0);
      ObjectSetText("BKGR", "g", 110, "Webdings", DarkViolet 	);
      ObjectSet("BKGR", OBJPROP_CORNER, 0);
      ObjectSet("BKGR", OBJPROP_BACK, TRUE);
      ObjectSet("BKGR", OBJPROP_XDISTANCE, 5);
      ObjectSet("BKGR", OBJPROP_YDISTANCE, 15);
   }
   if (ObjectFind("BKGR2") < 0) {
      ObjectCreate("BKGR2", OBJ_LABEL, 0, 0, 0);
      ObjectSetText("BKGR2", "g", 110, "Webdings", MidnightBlue);
      ObjectSet("BKGR2", OBJPROP_BACK, TRUE);
      ObjectSet("BKGR2", OBJPROP_XDISTANCE, 5);
      ObjectSet("BKGR2", OBJPROP_YDISTANCE, 60);
   }
   if (ObjectFind("BKGR3") < 0) {
      ObjectCreate("BKGR3", OBJ_LABEL, 0, 0, 0);
      ObjectSetText("BKGR3", "g", 110, "Webdings", MidnightBlue);
      ObjectSet("BKGR3", OBJPROP_CORNER, 0);
      ObjectSet("BKGR3", OBJPROP_BACK, TRUE);
      ObjectSet("BKGR3", OBJPROP_XDISTANCE, 5);
      ObjectSet("BKGR3", OBJPROP_YDISTANCE, 45);
   }
   if (ObjectFind("BKGR4") < 0) {
      ObjectCreate("BKGR4", OBJ_LABEL, 0, 0, 0);
      ObjectSetText("BKGR4", "g", 110, "Webdings", MidnightBlue);
      ObjectSet("BKGR4", OBJPROP_CORNER, 0);
      ObjectSet("BKGR4", OBJPROP_BACK, TRUE);
      ObjectSet("BKGR4", OBJPROP_XDISTANCE, 5);
      ObjectSet("BKGR4", OBJPROP_YDISTANCE, 84);
   }
}

//+--------------------------------------------------------------------------------------------------------------+
//| GetNameOP. ������� ���������� ������������ �������� ��������
//| ���������:
//|   op - ������������� �������� ��������
//+--------------------------------------------------------------------------------------------------------------+
string GetNameOP(int op) {
//+--------------------------------------------------------------------------------------------------------------+

	switch (op) {
		case OP_BUY      : return("Buy");
		case OP_SELL     : return("Sell");
		case OP_BUYLIMIT : return("Buy Limit");
		case OP_SELLLIMIT: return("Sell Limit");
		case OP_BUYSTOP  : return("Buy Stop");
		case OP_SELLSTOP : return("Sell Stop");
		default          : return("Unknown Operation");
	}
}

//+--------------------------------------------------------------------------------------------------------------+
//| ErrorDescription. ���������� �������� ������ �� � ������.
//+--------------------------------------------------------------------------------------------------------------+
string ErrorDescription(int error) {
//+--------------------------------------------------------------------------------------------------------------+

   string ErrorNumber;
   //---
   switch (error) {
   case 0:
   case 1:     ErrorNumber = "��� ������, �� ��������� ����������";                        break;
   case 2:     ErrorNumber = "����� ������";                                               break;
   case 3:     ErrorNumber = "������������ ���������";                                     break;
   case 4:     ErrorNumber = "�������� ������ �����";                                      break;
   case 5:     ErrorNumber = "������ ������ ����������� ���������";                        break;
   case 6:     ErrorNumber = "��� ����� � �������� ��������";                              break;
   case 7:     ErrorNumber = "������������ ����";                                          break;
   case 8:     ErrorNumber = "������� ������ �������";                                     break;
   case 9:     ErrorNumber = "������������ �������� ���������� ���������������� �������";  break;
   case 64:    ErrorNumber = "���� ������������";                                          break;
   case 65:    ErrorNumber = "������������ ����� �����";                                   break;
   case 128:   ErrorNumber = "����� ���� �������� ���������� ������";                      break;
   case 129:   ErrorNumber = "������������ ����";                                          break;
   case 130:   ErrorNumber = "������������ �����";                                         break;
   case 131:   ErrorNumber = "������������ �����";                                         break;
   case 132:   ErrorNumber = "����� ������";                                               break;
   case 133:   ErrorNumber = "�������� ���������";                                         break;
   case 134:   ErrorNumber = "������������ ����� ��� ���������� ��������";                 break;
   case 135:   ErrorNumber = "���� ����������";                                            break;
   case 136:   ErrorNumber = "��� ���";                                                    break;
   case 137:   ErrorNumber = "������ �����";                                               break;
   case 138:   ErrorNumber = "����� ���� - ������";                                        break;
   case 139:   ErrorNumber = "����� ������������ � ��� ��������������";                    break;
   case 140:   ErrorNumber = "��������� ������ �������";                                   break;
   case 141:   ErrorNumber = "������� ����� ��������";                                     break;
   case 145:   ErrorNumber = "����������� ���������, ��� ��� ����� ������� ������ � �����";break;
   case 146:   ErrorNumber = "���������� �������� ������";                                 break;
   case 147:   ErrorNumber = "������������� ���� ��������� ������ ��������� ��������";     break;
   case 148:   ErrorNumber = "���������� �������� � ���������� ������� �������� ������� "; break;
   //---- 
   case 4000:  ErrorNumber = "��� ������";                                                 break;
   case 4001:  ErrorNumber = "������������ ��������� �������";                             break;
   case 4002:  ErrorNumber = "������ ������� - ��� ���������";                             break;
   case 4003:  ErrorNumber = "��� ������ ��� ����� �������";                               break;
   case 4004:  ErrorNumber = "������������ ����� ����� ������������ ������";               break;
   case 4005:  ErrorNumber = "�� ����� ��� ������ ��� �������� ����������";                break;
   case 4006:  ErrorNumber = "��� ������ ��� ���������� ���������";                        break;
   case 4007:  ErrorNumber = "��� ������ ��� ��������� ������";                            break;
   case 4008:  ErrorNumber = "�������������������� ������";                                break;
   case 4009:  ErrorNumber = "�������������������� ������ � �������";                      break;
   case 4010:  ErrorNumber = "��� ������ ��� ���������� �������";                          break;
   case 4011:  ErrorNumber = "������� ������� ������";                                     break;
   case 4012:  ErrorNumber = "������� �� ������� �� ����";                                 break;
   case 4013:  ErrorNumber = "������� �� ����";                                            break;
   case 4014:  ErrorNumber = "����������� �������";                                        break;
   case 4015:  ErrorNumber = "������������ �������";                                       break;
   case 4016:  ErrorNumber = "�������������������� ������";                                break;
   case 4017:  ErrorNumber = "������ DLL �� ���������";                                    break;
   case 4018:  ErrorNumber = "���������� ��������� ����������";                            break;
   case 4019:  ErrorNumber = "���������� ������� �������";                                 break;
   case 4020:  ErrorNumber = "������ ������� ������������ ������� �� ���������";           break;
   case 4021:  ErrorNumber = "������������ ������ ��� ������, ������������ �� �������";    break;
   case 4022:  ErrorNumber = "������� ������";                                             break;
   case 4050:  ErrorNumber = "������������ ���������� ���������� �������";                 break;
   case 4051:  ErrorNumber = "������������ �������� ��������� �������";                    break;
   case 4052:  ErrorNumber = "���������� ������ ��������� �������";                        break;
   case 4053:  ErrorNumber = "������ �������";                                             break;
   case 4054:  ErrorNumber = "������������ ������������� �������-���������";               break;
   case 4055:  ErrorNumber = "������ ����������������� ����������";                        break;
   case 4056:  ErrorNumber = "������� ������������";                                       break;
   case 4057:  ErrorNumber = "������ ��������� ����������� ����������";                    break;
   case 4058:  ErrorNumber = "���������� ���������� �� ����������";                        break;
   case 4059:  ErrorNumber = "������� �� ��������� � �������� ������";                     break;
   case 4060:  ErrorNumber = "������� �� ������������";                                    break;
   case 4061:  ErrorNumber = "������ �������� �����";                                      break;
   case 4062:  ErrorNumber = "��������� �������� ���� string";                             break;
   case 4063:  ErrorNumber = "��������� �������� ���� integer";                            break;
   case 4064:  ErrorNumber = "��������� �������� ���� double";                             break;
   case 4065:  ErrorNumber = "� �������� ��������� ��������� ������";                      break;
   case 4066:  ErrorNumber = "����������� ������������ ������ � ��������� ����������";     break;
   case 4067:  ErrorNumber = "������ ��� ���������� �������� ��������";                    break;
   case 4099:  ErrorNumber = "����� �����";                                                break;
   case 4100:  ErrorNumber = "������ ��� ������ � ������";                                 break;
   case 4101:  ErrorNumber = "������������ ��� �����";                                     break;
   case 4102:  ErrorNumber = "������� ����� �������� ������";                              break;
   case 4103:  ErrorNumber = "���������� ������� ����";                                    break;
   case 4104:  ErrorNumber = "������������� ����� ������� � �����";                        break;
   case 4105:  ErrorNumber = "�� ���� ����� �� ������";                                    break;
   case 4106:  ErrorNumber = "����������� ������";                                         break;
   case 4107:  ErrorNumber = "������������ �������� ���� ��� �������� �������";            break;
   case 4108:  ErrorNumber = "�������� ����� ������";                                      break;
   case 4109:  ErrorNumber = "�������� �� ���������";                                      break;
   case 4110:  ErrorNumber = "������� ������� �� ���������";                               break;
   case 4111:  ErrorNumber = "�������� ������� �� ���������";                              break;
   case 4200:  ErrorNumber = "������ ��� ����������";                                      break;
   case 4201:  ErrorNumber = "��������� ����������� �������� �������";                     break;
   case 4202:  ErrorNumber = "������ �� ����������";                                       break;
   case 4203:  ErrorNumber = "����������� ��� �������";                                    break;
   case 4204:  ErrorNumber = "��� ����� �������";                                          break;
   case 4205:  ErrorNumber = "������ ��������� �������";                                   break;
   case 4206:  ErrorNumber = "�� ������� ��������� �������";                               break;
   case 4207:  ErrorNumber = "������ ��� ������ � ��������";                               break;
   default:    ErrorNumber = "����������� ������";
   }
   //---
   return (ErrorNumber);
}