import win32com.client
import csv

engine=win3com.client.Dispatch("OpenDSSEngine.DSS")
engine.Start("0")

engine.Text.Command='clear'
circuit = engine.ActiveCircuit
