/* Upstream's hasher behind one call, so that its context stays on this side
 * and the Rust side needs no copy of its layout. */

#include <stddef.h>

#include "sha1.h"

int sha1dc_oracle(const unsigned char *data, size_t len, int safe_hash,
		  int use_ubc, int reduced_round, unsigned char out[20])
{
	SHA1_CTX ctx;

	SHA1DCInit(&ctx);
	SHA1DCSetSafeHash(&ctx, safe_hash);
	SHA1DCSetUseUBC(&ctx, use_ubc);
	SHA1DCSetDetectReducedRoundCollision(&ctx, reduced_round);
	SHA1DCUpdate(&ctx, (const char *)data, len);
	return SHA1DCFinal(out, &ctx);
}
