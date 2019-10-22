#!/bin/bash

for i in {1..1}
do
    gnome-terminal -x bash -c './gpu_fv l 5 51717;bash' &
done
