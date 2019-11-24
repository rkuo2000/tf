#!/usr/bin/python3

# The MIT License (MIT)
#
# Copyright (c) 2015 Christian Zielinski
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULtAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import collections
import pickle
import sys
import os
import re

# Output file
output_file = "output/frequencies.html"

# File ending for frequency information
freq_file = ".freq"

# Number of characters to output
num_output_chars = 10000

# HTML template file
template_file = "nlp_template.html"

# Definition of Chinese characters
hanzi = '\u3007\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF'

if sys.maxunicode > 0xFFFF:
    hanzi += '\U00020000-\U0002A6DF\U0002A700-\U0002B73F'
    hanzi += '\U0002B740-\U0002B81F\U0002F800-\U0002FA1F'

re_hanzi = re.compile('[{}]'.format(hanzi))


def export_frequencies(chars_count, output_file):
    """
    Writes a frequency distribution to a HTML file.
    """
    with open(template_file, 'r', encoding='utf-8', errors='ignore') as f:
        template = f.read()

    chars_total = sum(chars_count.values())
    cum_freq = 0.0
    
    table = ""
    full_freq_data = enumerate(chars_count.most_common(num_output_chars), start=1)
    for rank, char_data in full_freq_data:
        char, count = char_data

        freq = 100.0 * count / chars_total
        cum_freq += freq

        data = (rank, char, freq, cum_freq)
        table += "<tr><td>{}</td><td>{}</td><td>{:.5f}</td><td>{:.5f}</td></tr>\n".format(*data)

    content = template.format(table=table)

    with open(output_file, 'w', encoding='utf-8', errors='ignore') as f:
        f.write(content)


def main():
    if len(sys.argv) != 2:
        print("Usage:\n> python3 {} input_directory/".format(sys.argv[0]))
        return 1

    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print("ERROR: <{}> is not a valid input directory".format(input_dir))
        return 2

    input_files = os.listdir(input_dir)
    input_files = [f for f in input_files if f.endswith(freq_file)]
    input_files = [os.path.join(input_dir, f) for f in input_files]

    chars_count = collections.Counter()
    num_files = len(input_files)

    for i in range(num_files):
        input_file = input_files[i]
        print("{}/{}: Processing file <{}> ...".format(i+1, num_files, input_file))

        with open(input_file, 'rb') as f:
            new_count = pickle.load(f, encoding='utf-8', errors='ignore')
        
        chars_count.update(new_count)

    export_frequencies(chars_count, output_file)
    print("Done.")


if __name__ == "__main__":
    main()
