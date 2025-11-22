# message(WARNING ">>> HASH ${SRC} to ${DST}")
file(SHA256 ${SRC} CHECKSUM)
file(WRITE ${DST} "#pragma once \n#define GEN_HASH \"${CHECKSUM}\"\n")