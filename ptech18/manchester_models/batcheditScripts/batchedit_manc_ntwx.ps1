$wd0 = pwd
# change this VVVV to the appropriate direction
$wd = "C:\Users\Matt\Desktop\batch_manc_ntwx" 

cd $wd
$mDss = $wd + "\Master.dss"
$myDss = $wd + "\Master_y.dss"

$Dir0 = dir -Directory
foreach ($Dir in $Dir0)
{
cd $Dir

$Dir1 = dir -Directory
foreach ($Fld in $Dir1)
{
cd $Fld

# Do stuff in the folders here:
Copy-Item $mDss
Copy-Item $myDss

$load_str = Get-Content .\Loads.txt
$load_str_copy = $load_str -replace ' Daily=Shape_\d*', ''
# ASCII seems to encode as utf8 rather than ascii ['utf8' argument encodes as utf8 BOM]:
$load_str_copy | out-file "Loads - Copy.txt" -encoding ASCII 

cd ..
}

cd ..
}


cd $wd0


# EXCEL modifying presently not working:
# trying to use https://www.itprotoday.com/sql-server/update-excel-spreadsheets-powershell
#$Excel = New-Object -Com Excel.Application
#$pwd = pwd
#$FilePath = Join-Path $pwd "\XY_Position.xls"
#$Workbook = $Excel.Workbooks.Open($FilePath)
#$ws = $Workbook.worksheets | where-object {$_.Name -eq "Sheet1"}
