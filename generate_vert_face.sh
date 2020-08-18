#!/bin/bash
fileName=$1
Z=$(grep -n -m 1 "NOD" "$fileName".msh |sed  's/\([0-9]*\).*/\1/')
W=$(grep -n -m 1 "ENDNOD" "$fileName".msh |sed  's/\([0-9]*\).*/\1/')
< "$fileName".msh tail -n +"$((Z+2))" | head -n "$((W - Z - 2))" > vert.txt
Y=$(grep -n -m 1 "ENDELM" "$fileName".msh |sed  's/\([0-9]*\).*/\1/')
awk '{print NF}' "$fileName".msh > tmp.txt
X=$(grep -n -m 1 "8" tmp.txt | cut -f1 -d:)
< "$fileName".msh tail -n +"$X" | head -n "$((Y - X))" > face.txt
rm tmp.txt