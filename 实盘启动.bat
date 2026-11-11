@echo off
title Binance DCA Live Bot
cd /d "%~dp0"
echo Starting Binance spot DCA bot ...
echo (Log file: logs\live_trade_real.txt)
python binance_live_bot.py
echo.
echo Bot stopped. Press any key to close.
pause >nul
