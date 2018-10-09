function BRX = bracket_c(X)

	BRX = [real(X), imag(X); imag(X), -real(X)];

