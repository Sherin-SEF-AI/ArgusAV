"""Strict MCAP reader for the tests (stdlib only): checks CRCs, the summary, and that every
chunk index and message index points at the record it claims. Not used by avrig itself."""
import struct
import zlib

MAGIC = b"\x89MCAP0\r\n"


def _str(b, i):
    n = struct.unpack_from("<I", b, i)[0]
    return b[i + 4:i + 4 + n].decode(), i + 4 + n


def _map(b, i, kfmt=None, vfmt=None):
    n = struct.unpack_from("<I", b, i)[0]
    j, end, out = i + 4, i + 4 + n, {}
    while j < end:
        if kfmt:
            k, v = struct.unpack_from("<" + kfmt + vfmt, b, j)
            j += struct.calcsize("<" + kfmt + vfmt)
        else:
            k, j = _str(b, j)
            v, j = _str(b, j)
        out[k] = v
    return out, end


def records(b, i, end):
    while i < end:
        op, n = struct.unpack_from("<BQ", b, i)
        yield i, op, b[i + 9:i + 9 + n]
        i += 9 + n
    assert i == end


def read(path):
    """-> dict(channels={id: (topic, schema, enc, meta)}, schemas={id: (name, enc, text)},
    messages=[(topic, log_time, data)], stats, chunks, metadata)."""
    b = open(path, "rb").read()
    assert b[:8] == MAGIC and b[-8:] == MAGIC
    fo = len(b) - 8 - 29
    op, n, s_start, so_start, s_crc = struct.unpack_from("<BQQQI", b, fo)
    assert (op, n) == (0x02, 20)
    assert s_crc == zlib.crc32(b[s_start:fo + 25])
    schemas, channels, msgs, metadata, chunk_idx, stats = {}, {}, [], {}, [], None
    data_end = None
    for at, op, body in records(b, 8, s_start):
        if op == 0x0F:
            assert struct.unpack("<I", body)[0] == zlib.crc32(b[:at])
            data_end = at
        elif op == 0x0C:
            name, j = _str(body, 0)
            metadata[name] = _map(body, j)[0]
        elif op == 0x06:
            t0, t1, usize, ucrc = struct.unpack_from("<QQQI", body)
            comp, j = _str(body, 28)
            clen = struct.unpack_from("<Q", body, j)[0]
            recs = body[j + 8:j + 8 + clen]
            assert comp == "" and usize == clen and zlib.crc32(recs) == ucrc
            for _, rop, rb in records(recs, 0, len(recs)):
                if rop == 0x03:
                    sid = struct.unpack_from("<H", rb)[0]
                    name, k = _str(rb, 2)
                    enc, k = _str(rb, k)
                    schemas[sid] = (name, enc, rb[k + 4:].decode())
                elif rop == 0x04:
                    cid, sid = struct.unpack_from("<HH", rb)
                    topic, k = _str(rb, 4)
                    enc, k = _str(rb, k)
                    channels[cid] = (topic, sid, enc, _map(rb, k)[0])
                elif rop == 0x05:
                    cid, seq, lt, pt = struct.unpack_from("<HIQQ", rb)
                    assert t0 <= lt <= t1
                    msgs.append((channels[cid][0], lt, rb[22:]))
    assert data_end is not None
    for at, op, body in records(b, s_start, fo):
        if op == 0x08:
            t0, t1, cstart, clen = struct.unpack_from("<QQQQ", body)
            offs, j = _map(body, 32, "H", "Q")
            assert b[cstart] == 0x06 and 9 + struct.unpack_from("<Q", b, cstart + 1)[0] == clen
            chunk = b[cstart + 9:cstart + clen]
            comp, k = _str(chunk, 28)
            recs = chunk[k + 8:]
            for cid, mo in offs.items():          # every index entry lands on its message
                assert b[mo] == 0x07
                n = struct.unpack_from("<Q", b, mo + 1)[0]
                ib = b[mo + 9:mo + 9 + n]
                assert struct.unpack_from("<H", ib)[0] == cid
                ents = ib[6:]
                for e in range(0, len(ents), 16):
                    lt, ro = struct.unpack_from("<QQ", ents, e)
                    assert recs[ro] == 0x05
                    mcid, _, mlt = struct.unpack_from("<HIQ", recs, ro + 9)
                    assert (mcid, mlt) == (cid, lt)
            chunk_idx.append((t0, t1, cstart, clen))
        elif op == 0x0B:
            mc, sc, cc, ac, mdc, chc, ts, te = struct.unpack_from("<QHIIIIQQ", body)
            counts = _map(body, 42, "H", "Q")[0]
            stats = dict(messages=mc, schemas=sc, channels=cc, metadata=mdc, chunks=chc,
                         start=ts, end=te, counts=counts)
        elif op == 0x0E:
            gop, gstart, glen = struct.unpack("<BQQ", body)
            assert gop in {o for _, o, _ in records(b, gstart, gstart + glen)}
    assert stats and stats["messages"] == len(msgs) and stats["chunks"] == len(chunk_idx)
    return dict(channels=channels, schemas=schemas, messages=msgs, stats=stats,
                chunks=chunk_idx, metadata=metadata)
