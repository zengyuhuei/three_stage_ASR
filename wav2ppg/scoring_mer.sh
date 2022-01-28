#!/bin/bash


out=out.file
ref=ref.file
result_file=mixed_error_rate

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh || exit 1;

files=()
files+=($out)
files+=($ref)


echo files
for f in "${files[@]}" ; do
  cut -d ' ' -f 1 $f > $f.ids
  cut -d ' ' -f 2- $f > $f.text
  cat $f.text | sed -Ee "s/([A-Za-z']+)/ \0 /g" | \
    sed -Ee "s/(\[|\<)\s([A-Za-z']+)\s(\]|\>)/ \1\2\3 /g" | perl -CSDA -ane '
    {
      #print $F[0];
      foreach $s (@F[0..$#F]) {
        if (($s =~ /\[.*\]/) || ($s =~ /\<.*\>/) || ($s =~ "!SIL")) {
          print " $s";
        } elsif (($s =~ /[A-Za-z\.'\'']+/)) {
          print " $s";
        } else {
          @chars = split "", $s;
          foreach $c (@chars) {
            print " $c";
          }
        }
      }
      print "\n";
    }' | \
  paste -d ' ' $f.ids - > $f.mixed.txt
  rm $f.ids $f.text
done
# case of first column is id
# print $F[0];
# foreach $s (@F[1..$#F]) {

cat $out.mixed.txt | compute-wer --text --mode=present ark:$ref.mixed.txt ark,p:- > $result_file

cat $out.mixed.txt | align-text --special-symbol="'***'" ark:$ref.mixed.txt ark:- ark,t:- | \
  utils/scoring/wer_per_utt_details.pl --special-symbol "'***'" > $result_file.per_utt

