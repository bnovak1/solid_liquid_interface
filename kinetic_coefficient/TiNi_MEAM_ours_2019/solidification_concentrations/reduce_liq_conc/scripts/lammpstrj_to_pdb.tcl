package require pbctools

proc lammpstrj_to_pdb {lmpin pdbout element1 element2} {

    set basemol [mol new $lmpin waitfor all]

    pbc wrap

    set sel [atomselect $basemol "type 1"]
    $sel set name $element1
    $sel delete

    set sel [atomselect $basemol "type 2"]
    $sel set name $element2
    $sel delete

    animate write pdb $pdbout beg 0 end 0 $basemol

}
