#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netdb.h>

#define PORT      4242
#define REC_END   "\r\n.\r\n"   /* daemon record terminator */
#define REC_END_LEN 6

int main(int argc __attribute__((unused)), char *argv[] __attribute__((unused))) {
	int sock;
	struct sockaddr_in server_addr;
	char buf[BUFSIZ];
	char msg[BUFSIZ];

	sock = socket(AF_INET, SOCK_STREAM, 0);
	if (sock < 0) {
		perror("socket");
		return 1;
	}

	memset(&server_addr, 0, sizeof(server_addr));
	server_addr.sin_family = AF_INET;
	server_addr.sin_port   = htons(PORT);
	server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");

	if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
		perror("connect");
		close(sock);
		return 1;
	}

	int mlen = snprintf(msg, sizeof(msg), "chat\n");
	if (send(sock, msg, mlen, 0) != mlen) {
		perror("send");
		return 1;
	}

	setvbuf(stdout, NULL, _IONBF, 0);
	printf("Connected! Type your prompts (empty line to quit).\n\n");

	int tty = isatty(STDIN_FILENO) && isatty(STDOUT_FILENO);

	while (1) {
		char pend[REC_END_LEN - 1];
		size_t npend = 0;
		int done = 0;

		if (tty)
			printf("> ");

		if (!fgets(buf, sizeof(buf), stdin))
			break;

		size_t len = strlen(buf);
		if (len == 0 || buf[0] == '\n')
			break;

		if (buf[len - 1] == '\n')
			buf[len - 1] = '\0';

		int mlen = snprintf(msg, sizeof(msg), "ask %s\n", buf);

		if ((size_t) mlen >= sizeof(msg)) {
			fprintf(stderr, "Prompt too long\n");
			continue;
		}

		if (send(sock, msg, mlen, 0) != mlen) {
			perror("send");
			break;
		}

		while (!done) {
			ssize_t cn = read(sock, buf, sizeof(buf));

			if (cn <= 0)
				break;

			size_t total = npend + (size_t) cn;
			char *comb = malloc(total + 1);
			if (!comb)
				break;

			memcpy(comb, pend, npend);
			memcpy(comb + npend, buf, (size_t) cn);
			comb[total] = '\0';

			char *match = strstr(comb, REC_END);
			if (match) {
				fwrite(comb, 1, (size_t)(match - comb), stdout);
				done = 1;
			} else {
				size_t safe = total >= REC_END_LEN - 1
				    ? total - (REC_END_LEN - 1) : 0;

				fwrite(comb, 1, safe, stdout);
				npend = total - safe;
				memcpy(pend, comb + safe, npend);
			}

			free(comb);
		}

		putchar('\n');

		if (!done) {
			fprintf(stderr,
			    "Connection closed before reply completed\n");
			break;
		}
	}

	close(sock);
	printf("Session closed.\n");
	return 0;
}