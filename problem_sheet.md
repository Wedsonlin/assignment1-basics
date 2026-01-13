# Problem Sheet



## Problem (unicode1)

#### (a) What Unicode character does chr(0) return?

It returns a null character `'\x00'`.

#### (b) How does this character’s string representation (\__repr__()) differ from its printed representation?
\__repr__() shows the non-escape representation of a character's string.

````
a = chr(0)
print(a) # just show nothing
print(a.__repr__()) # show '\x00'
````

#### (c) What happens when this character occurs in text?

It does exist in text and counts for length, but isn't visiable in regular print statment. 



## Problem (unicode2)

#### (a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32?

1. Most real-world text is already UTF-8.
2. Much shorter byte sequences for ASCII-heavy data. 
3. UTF-16 or UTF-32 introduces tons of `0x00` bytes, which reduce tokenizer's learning ability.

#### (b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.
````python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
	return "".join([bytes([b]).decode("utf-8") for b in bytestring])
````

Because to most of the characters, its UTF-8 encode is multi-byte.  

An example: `'你是谁？'`

#### (c) Give a two byte sequence that does not decode to any Unicode character(s).

`b'\x00\xd8'`



