set basemol [mol new five.37600.lammpstrj]

set infile [open "five.37600.lammpstrj_index_liquid.dat" r]
set liquid_ind [gets $infile]
close $infile
set liquid [atomselect $basemol "index $liquid_ind"]
$liquid set type 2
$liquid delete

set infile [open "five.37600.lammpstrj_index_interface.dat" r]
set interface_ind [gets $infile]
close $infile
set interface [atomselect $basemol "index $interface_ind"]
$interface set type 3
$interface delete

pbc wrap
pbc wrap -center com -centersel "type 2"

set sel [atomselect $basemol all]
set box_center [expr {[lindex [lindex [pbc get] 0] 2]/2.0}]
set center [lindex [measure center $sel] 2]
set dz [expr {$box_center - $center}]
$sel moveby "0 0 $dz"
