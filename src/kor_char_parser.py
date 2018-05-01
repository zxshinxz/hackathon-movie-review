# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

cho = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"  # len = 19
jung = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"  # len = 21
# len = 27
jong = "ㄱ/ㄲ/ㄱㅅ/ㄴ/ㄴㅈ/ㄴㅎ/ㄷ/ㄹ/ㄹㄱ/ㄹㅁ/ㄹㅂ/ㄹㅅ/ㄹㅌ/ㄹㅍ/ㄹㅎ/ㅁ/ㅂ/ㅂㅅ/ㅅ/ㅆ/ㅇ/ㅈ/ㅊ/ㅋ/ㅌ/ㅍ/ㅎ".split(
    '/')
test = cho + jung + ''.join(jong)

hangul_length = len(cho) + len(jung) + len(jong)  # 67


def is_valid_decomposition_atom(x):
    return x in test


def decompose(x):
    in_char = x
    if x < ord('가') or x > ord('힣'):
        return chr(x)
    x = x - ord('가')
    y = x // 28
    z = x % 28
    x = y // 21
    y = y % 21
    # if there is jong, then is z > 0. So z starts from 1 index.
    zz = jong[z - 1] if z > 0 else ''
    if x >= len(cho):
        print('Unknown Exception: ', in_char, chr(in_char), x, y, z, zz)
    return cho[x] + jung[y] + zz

# 단자음 14개 (ㄱ ㄴ ㄷ ㄹ ㅁ ㅂ ㅅ ㅇ ㅈ ㅊ ㅋ ㅌ ㅎ)
# 쌍자음 5개 (ㄲ ㄸ ㅃ ㅆ ㅉ)
# 복자음 11개 (ㄳ ㄵ ㄶ ㄺ ㄻ ㄼ ㄽ ㄾ ㄿ ㅀ ㅄ)
# 단모음 8개 (ㅏ ㅓ ㅗ ㅜ ㅡ ㅣ ㅐ ㅔ)
# 이중모음 6개 (ㅑ ㅕ ㅛ ㅠ ㅒ ㅖ)
# 복모음 7개 (ㅘ ㅙ ㅚ ㅝ ㅞ ㅟ ㅢ)
# 총 51 [195,245]: hangul danja,danmo

# 한글 음절은 초성 자음과 중성 모음, 그리고 선택적 종성 자음으로 이루어져 있다.
# 초성 자음은 총 19개 (단자음 14 + 쌍자음 5)이고,
# 중성 모음은 총 21개 (단모음 8 + 이중모음 6 + 복모음 7)이고
# 종성 자음은 총 27개 (단자음 14개 + 쌍자음 2(ㄲㅆ) + 복자음 11)이다.
#
# 따라서 한글의 총 음절 수는 11,172자 (19 * 21 * (27 + 1)) 이다.
# (+1은 종성이 없는 경우)
#


def decompose_as_one_hot(in_char, warning=True):
    one_hot = []
    # print(ord('ㅣ'), chr(0xac00))
    # [0,66]: hangul / [67,194]: ASCII / [195,245]: hangul danja,danmo / [246,249]: special characters
    # Total 250 dimensions.

    # 한글만 포함
    if ord('가') <= in_char <= ord('힣'):  # 가:44032 , 힣: 55203Ω
        x = in_char - 44032  # in_char - ord('가')
        y = x // 28   # 19 * 21 * (27 + 1) 에서  (27 + 1) < cancel (초성 중성만 남음)
        z = x % 28    # 19 * 21 * (27 + 1) 에서  (27 + 1) 의 나머지는 종성
        x = y // 21   # 19 * 21 에서 21 < cancel (초성만 남음)
        y = y % 21    # 19 * 21 에서 21 의 나머지는 중성

        # x = 초성 y = 중성 z = 종성

        # if there is jong, then is z > 0. So z starts from 1 index.
        zz = jong[z - 1] if z > 0 else ''
        # 초성 길이 보다 x 가 크면 모르는 글자
        if x >= len(cho):
            if warning:
                print('Unknown Exception: ', in_char,
                      chr(in_char), x, y, z, zz)

        # 초성 중성 종성을 순차적으로 vector화 시켜서 넣는다
        # cho = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ" + jung = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ" + jong = "ㄱ/ㄲ/ㄱㅅ/ㄴ/ㄴㅈ/ㄴㅎ/ㄷ/ㄹ/ㄹㄱ/ㄹㅁ/ㄹㅂ/ㄹㅅ/ㄹㅌ/ㄹㅍ/ㄹㅎ/ㅁ/ㅂ/ㅂㅅ/ㅅ/ㅆ/ㅇ/ㅈ/ㅊ/ㅋ/ㅌ/ㅍ/ㅎ".split('/')
        # 총 0,66
        one_hot.append(x)
        one_hot.append(len(cho) + y)
        # 종성은 있을시만 넣음
        if z > 0:
            one_hot.append(len(cho) + len(jung) + (z - 1))
        return one_hot
    else:
        if in_char < 128:
            result = hangul_length + in_char  # 67~
        elif ord('ㄱ') <= in_char <= ord('ㅣ'):
            # 194~ # [ㄱ:12593]~[ㅣ:12643] (len = 51)
            result = hangul_length + 128 + (in_char - 12593)
        elif in_char == ord('♡'):
            result = hangul_length + 128 + 51  # 245~ # ♡
        elif in_char == ord('♥'):
            result = hangul_length + 128 + 51 + 1  # ♥
        elif in_char == ord('★'):
            result = hangul_length + 128 + 51 + 2  # ★
        elif in_char == ord('☆'):
            result = hangul_length + 128 + 51 + 3  # ☆
        else:
            if warning:
                print('Unhandled character:', chr(in_char), in_char)
            # unknown character
            result = hangul_length + 128 + 51 + 4  # for unknown character

        return [result]


def decompose_str(string):
    return ''.join([decompose(ord(x)) for x in string])


def decompose_str_as_one_hot(string, warning=True):
    tmp_list = []
    for x in string:
        da = decompose_as_one_hot(ord(x), warning=warning)
        tmp_list.extend(da)
    return tmp_list
