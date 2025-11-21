class GPT4KnowledgeGenerator:
    def __init__(self, api_key=None, use_proxy=True):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.use_proxy = use_proxy
        self.cache = {}
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        self.proxies = {
            "http": "http://127.0.0.1:7890",
            "https": "http://127.0.0.1:7890"
        } if use_proxy else None

    def generate_sentence_expansion(self, sentence, beam_size=4):
        cache_key = f"exp_{sentence}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = f"Expand this sentence with more details while keeping the core meaning: {sentence}"

        try:
            payload = json.dumps({
                "model": "GPT-4 Turbo",
                "messages": [
                    {"role": "system",
                     "content": "You are a helpful assistant that expands sentences with relevant details."},
                    {"role": "user", "content": prompt}
                ],
                "n": beam_size,
                "max_tokens": 64,
                "temperature": 0.7
            })

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=self.headers,
                data=payload,
                proxies=self.proxies,
                timeout=30
            )

            if response.status_code == 200:
                expansions = []
                for choice in response.json()['choices']:
                    expansions.append(choice['message']['content'].strip())
                self.cache[cache_key] = expansions[:beam_size]
                return expansions[:beam_size]
            else:
                return self._fallback_expansion(sentence, beam_size)

        except Exception:
            return self._fallback_expansion(sentence, beam_size)

    def generate_aspect_knowledge(self, sentence, aspect, beam_size=4):
        cache_key = f"asp_{sentence}_{aspect}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = f"Given the sentence: '{sentence}', provide knowledge about {aspect}"

        try:
            payload = json.dumps({
                "model": "GPT-4 Turbo",
                "messages": [
                    {"role": "system",
                     "content": "You are a helpful assistant that provides contextual knowledge about aspects."},
                    {"role": "user", "content": prompt}
                ],
                "n": beam_size,
                "max_tokens": 64,
                "temperature": 0.7
            })

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=self.headers,
                data=payload,
                proxies=self.proxies,
                timeout=30
            )

            if response.status_code == 200:
                knowledge = []
                for choice in response.json()['choices']:
                    knowledge.append(choice['message']['content'].strip())
                self.cache[cache_key] = knowledge[:beam_size]
                return knowledge[:beam_size]
            else:
                return self._fallback_aspect_knowledge(aspect, beam_size)

        except Exception:
            return self._fallback_aspect_knowledge(aspect, beam_size)

    def _fallback_expansion(self, sentence, beam_size):
        expansions = []
        expansions.append(f"This elaborates on: {sentence}")
        expansions.append(f"In detail, {sentence} This provides context.")
        expansions.append(f"Specifically, {sentence} The meaning is clear.")
        expansions.append(f"To explain, {sentence} This shows the picture.")
        return expansions[:beam_size]

    def _fallback_aspect_knowledge(self, aspect, beam_size):
        knowledge = []
        knowledge.append(f"The {aspect} is a key element.")
        knowledge.append(f"Regarding {aspect}, it has importance.")
        knowledge.append(f"The {aspect} feature is noteworthy.")
        knowledge.append(f"About {aspect}, multiple factors apply.")
        return knowledge[:beam_size]