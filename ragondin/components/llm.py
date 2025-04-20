import httpx
import json
import copy


class LLM:
    def __init__(self, llm_config, logger=None):
        self.logger = logger
        default_llm_config = dict(llm_config)
        self._api_key = default_llm_config.pop("api_key", None)
        self._base_url = default_llm_config.pop("base_url", None)
        self.default_llm_config = default_llm_config

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    async def completions(self, request: dict):
        ragondin_model = request.pop("model", None)
        payload = copy.deepcopy(self.default_llm_config)
        payload.update(request)

        timeout = httpx.Timeout(4 * 10)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url=f"{self._base_url}completions", headers=self.headers, json=payload
            )
            response.raise_for_status()

            data = response.json()
            data["model"] = ragondin_model
            yield data

    async def chat_completion(self, request: dict):
        ragondin_model = request.pop("model", None)
        payload = copy.deepcopy(self.default_llm_config)
        payload.update(request)

        stream = request["stream"]

        timeout = httpx.Timeout(4 * 60)
        async with httpx.AsyncClient(timeout=timeout) as client:
            if stream:
                try:
                    async with client.stream(
                        "POST",
                        url=f"{self._base_url}chat/completions",
                        headers=self.headers,
                        json=payload,
                    ) as response:
                        async for line in response.aiter_lines():
                            if line.startswith("data:"):
                                if "[DONE]" in line:
                                    yield f"{line}\n\n"
                                else:
                                    try:
                                        data_str = line[len("data: ") :]
                                        data = json.loads(data_str)
                                        data["model"] = ragondin_model
                                        new_line = f"data: {json.dumps(data)}\n\n"
                                        yield new_line
                                    except json.JSONDecodeError as e:
                                        raise e

                except Exception as e:
                    raise e

            else:  # Handle non-streaming response
                try:
                    response = await client.post(
                        url=f"{self._base_url}chat/completions",
                        headers=self.headers,
                        json=payload,
                    )
                    response.raise_for_status()

                    data = response.json()  # Modify the fields here
                    data["model"] = ragondin_model
                    yield data
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in API response: {str(e)}")
