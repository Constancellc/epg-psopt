function RGC = set_taps( RGC )

ii = RGC.First;
while ii
    RGC.MaxTapChange = 0;
    ii = RGC.next;
end

end