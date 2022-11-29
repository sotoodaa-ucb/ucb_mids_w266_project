import re

from nltk.stem import WordNetLemmatizer


class Preprocessor:
    def __init__(self) -> None:
        self.stemmer = WordNetLemmatizer()

    def links_to_word(self, text):
        return re.sub("https?:\/\/[^\s]+", " link ", text)

    def no_char(self, text):
        text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)
        text = re.sub(r"\^[a-zA-Z]\s+", " ", text)
        text = re.sub(r"\s+[a-zA-Z]$", " ", text)
        return text

    def no_html_tags(self, text):
        return re.sub("<.*?>", " ", text)

    def no_multi_spaces(self, text):
        return re.sub(r"\s+", " ", text, flags=re.I)

    def lemmatize(self, text):
        tokens = text.split()
        tokens = [self.stemmer.lemmatize(word) for word in tokens]
        return " ".join(tokens)

    def underscore_to_space(self, text: str):
        text = text.replace("_", " ")
        text = text.replace("-", " ")
        return text

    def no_markdown_special(self, text: str):
        try:
            text = text[0] + re.sub(r"(?<!\n)[\*\+\-\>]", " ", text[1:])
            text = re.sub(r"\(\)\[\]\{\}\<\>\~\|\`\.", " ", text)
        except IndexError:
            return ''
        return text

    def code_preprocess(self, code):
        code = self.links_to_word(code)
        code = self.lemmatize(code)
        return code

    def markdown_preprocess(self, code: str):
        """
        1. Replace new lines with unused token.
        2. Remove HTML Tags and special markdown symbols.
        3. Clear html tags first, then markdown...
        """
        code = code.replace("\n", "[unused1]")
        code = self.links_to_word(code)
        code = self.no_html_tags(code)
        code = self.no_markdown_special(code)
        code = self.no_multi_spaces(code)
        code = self.lemmatize(code)
        return code

    def preprocessor(self, text: str, cell_type: str):
        return dict(code=self.code_preprocess, markdown=self.markdown_preprocess)[cell_type](text)
