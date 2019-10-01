#!/bin/bash

for i in {1..13}
do
    gnome-terminal -x bash -c './gpu_fv s l 5 51717;bash' &
done
