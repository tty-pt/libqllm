/* openai_chat.h — OpenAI-compatible chat completions endpoint for qllmd. */

#ifndef OPENAI_CHAT_H
#define OPENAI_CHAT_H

int openai_chat_init(const char *model_path);
void openai_chat_shutdown(void);

#endif /* OPENAI_CHAT_H */