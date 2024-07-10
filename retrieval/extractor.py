import mwparserfromhell as mwp
import re

# we don't want to include these sections in the final text
ENDING_PHRASES = [
  "Reference",
  "References",
  "Notes",
  "Bibliography",
  "Reflist", 
  "Reference list",
  "Footnote",
  "Footnotes",
  "See also",
  "See Also",
  "Gallery",
  "External links",
  "External Links",
  "Links",
  "Further reading",
  "Further Reading",
  "Sources",
  "Notes",
  "Bibliography",
  "Selected bibliography",
  "References and notes",
  "References:",
  "References and external links",
  "ReferencesSources",
  "References Sources",
  "== References ==",
  # Can't do `Works`, it destroys some of the Wikipedia formatting and also applies to artists
]


def reverse_in(text, substrings):
    """Checks if any of the substrings are in the text."""
    return any(substring in text for substring in substrings)


def _parse_and_clean_wikicode(raw_content):
    """Strips formatting and unwanted sections from raw page content."""
    wikicode = mwp.parse(raw_content)

    def rm_wikilink(obj):
        return str(obj.title).lower().startswith(("file:", "image:", "media:"))

    def rm_tag(obj):
        return str(obj.tag).lower() in {"ref", "table", "tbody", "tr", "td", "th", "li", "ul", "infobox"}

    def rm_template(obj):
        return obj.name.lower() in {
            "reflist", "notelist", "notelist-ua", "notelist-lr", "notelist-ur", "notelist-lg",
            "cite", "cite web", "cite news", "cite book", "cite journal", "infobox"
        }

    def try_remove_obj(obj, section):
        try:
            section.remove(obj)
        except ValueError:
            pass

    section_text = []
    for section in wikicode.get_sections(flat=True, include_lead=True, include_headings=True):
        if section.filter_headings():
            heading = str(section.filter_headings()[0].title).strip()
            if reverse_in(heading, ENDING_PHRASES):
                break

        # Remove unwanted elements
        for obj in section.ifilter_wikilinks(matches=rm_wikilink, recursive=True):
            try_remove_obj(obj, section)
        for obj in section.ifilter_templates(matches=rm_template, recursive=True):
            try_remove_obj(obj, section)
        for obj in section.ifilter_tags(matches=rm_tag, recursive=True):
            try_remove_obj(obj, section)

        # Remove all references, including <ref> tags
        for ref in section.filter_tags(matches=lambda tag: tag.tag.lower() == 'ref'):
            try_remove_obj(ref, section)

        # Remove Wikipedia-style tables
        for table in section.filter_templates(matches=lambda t: t.name.strip().lower() == 's-start'):
            try_remove_obj(table, section)

        # Clean remaining text
        clean_text = section.strip_code()

        # Remove remaining table-like structures
        clean_lines = []
        in_table = False
        for line in clean_text.split('\n'):
            if line.strip().startswith('{|') or line.strip().startswith('|-'):
                in_table = True
            elif line.strip().startswith('|}'):
                in_table = False
            elif not in_table and not line.strip().startswith('|'):
                clean_lines.append(line)

        clean_text = '\n'.join(clean_lines)

        # if the above parsing missed any, just remove them
        clean_text = re.sub(r'<[^>]+>', '', clean_text)
        # get cite and infobox templates
        clean_text = re.sub(r'{{cite.*?}}', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
        clean_text = re.sub(r'{{infobox.*?}}', '', clean_text, flags=re.DOTALL | re.IGNORECASE)

        # Remove empty lines and excessive whitespace
        clean_lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
        clean_text = '\n'.join(clean_lines)

        # remove any {{.*}} if they close
        clean_text = re.sub(r'{{.*}}', '', clean_text, flags=re.DOTALL)
        # if the strings "{{" is in there with no "}}" then remove {{ and the word attached to it
        if "{{" in clean_text or "}}" in clean_text:
            clean_text_words = clean_text.split()
            index_of_open = [idx for idx, word in enumerate(clean_text_words) if ("{{" in word or "}}" in word)]
            # delete the word that has the {{
            for i in reversed(index_of_open):
                clean_text_words.pop(i)
            # put it back together
            clean_text = " ".join(clean_text_words)

        if clean_text:
            section_text.append(clean_text)

    final_str = "\n\n".join(section_text)
    return final_str