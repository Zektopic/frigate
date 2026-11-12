//! Minimal JSON writer, so the crate needs no dependencies.
//!
//! Only what the collector emits: objects, arrays, strings, integers, finite
//! floats, booleans. Commas are inserted automatically.

pub struct Json {
    buf: String,
    /// One entry per open container: whether it already holds a value.
    stack: Vec<bool>,
    /// A key was just written, so the next value follows it directly.
    after_key: bool,
}

impl Json {
    pub fn new() -> Self {
        Json {
            buf: String::with_capacity(4096),
            stack: Vec::new(),
            after_key: false,
        }
    }

    pub fn finish(self) -> String {
        debug_assert!(self.stack.is_empty(), "unbalanced JSON containers");
        self.buf
    }

    fn before_value(&mut self) {
        if self.after_key {
            self.after_key = false;
            return;
        }
        if let Some(has_value) = self.stack.last_mut() {
            if *has_value {
                self.buf.push(',');
            }
            *has_value = true;
        }
    }

    pub fn begin_object(&mut self) -> &mut Self {
        self.before_value();
        self.buf.push('{');
        self.stack.push(false);
        self
    }

    pub fn end_object(&mut self) -> &mut Self {
        self.stack.pop();
        self.buf.push('}');
        self
    }

    pub fn begin_array(&mut self) -> &mut Self {
        self.before_value();
        self.buf.push('[');
        self.stack.push(false);
        self
    }

    pub fn end_array(&mut self) -> &mut Self {
        self.stack.pop();
        self.buf.push(']');
        self
    }

    pub fn key(&mut self, key: &str) -> &mut Self {
        self.before_value();
        write_string(&mut self.buf, key);
        self.buf.push(':');
        self.after_key = true;
        self
    }

    pub fn string(&mut self, value: &str) -> &mut Self {
        self.before_value();
        write_string(&mut self.buf, value);
        self
    }

    pub fn uint(&mut self, value: u64) -> &mut Self {
        self.before_value();
        self.buf.push_str(&value.to_string());
        self
    }

    pub fn int(&mut self, value: i64) -> &mut Self {
        self.before_value();
        self.buf.push_str(&value.to_string());
        self
    }

    pub fn bool(&mut self, value: bool) -> &mut Self {
        self.before_value();
        self.buf.push_str(if value { "true" } else { "false" });
        self
    }

    /// A float rounded to `decimals` places. JSON has no NaN or infinity, so
    /// those are written as `null`.
    pub fn float(&mut self, value: f64, decimals: usize) -> &mut Self {
        self.before_value();
        if value.is_finite() {
            self.buf.push_str(&format!("{value:.decimals$}"));
        } else {
            self.buf.push_str("null");
        }
        self
    }

    // Keyed shorthands.

    pub fn field_str(&mut self, key: &str, value: &str) -> &mut Self {
        self.key(key).string(value)
    }

    pub fn field_uint(&mut self, key: &str, value: u64) -> &mut Self {
        self.key(key).uint(value)
    }

    pub fn field_int(&mut self, key: &str, value: i64) -> &mut Self {
        self.key(key).int(value)
    }

    pub fn field_bool(&mut self, key: &str, value: bool) -> &mut Self {
        self.key(key).bool(value)
    }

    pub fn field_float(&mut self, key: &str, value: f64, decimals: usize) -> &mut Self {
        self.key(key).float(value, decimals)
    }

    /// Writes the field only when the value is known.
    pub fn field_opt_float(&mut self, key: &str, value: Option<f64>, decimals: usize) -> &mut Self {
        if let Some(v) = value {
            self.field_float(key, v, decimals);
        }
        self
    }

    pub fn field_opt_uint(&mut self, key: &str, value: Option<u64>) -> &mut Self {
        if let Some(v) = value {
            self.field_uint(key, v);
        }
        self
    }
}

fn write_string(buf: &mut String, value: &str) {
    buf.push('"');
    for c in value.chars() {
        match c {
            '"' => buf.push_str("\\\""),
            '\\' => buf.push_str("\\\\"),
            '\n' => buf.push_str("\\n"),
            '\r' => buf.push_str("\\r"),
            '\t' => buf.push_str("\\t"),
            c if (c as u32) < 0x20 => buf.push_str(&format!("\\u{:04x}", c as u32)),
            c => buf.push(c),
        }
    }
    buf.push('"');
}

#[cfg(test)]
mod tests {
    use super::Json;

    #[test]
    fn nested_containers_get_commas() {
        let mut j = Json::new();
        j.begin_object();
        j.field_uint("a", 1);
        j.key("b").begin_array();
        j.uint(1).uint(2);
        j.begin_object().field_str("c", "d").end_object();
        j.end_array();
        j.key("e").begin_object().end_object();
        j.field_bool("f", true);
        j.end_object();
        assert_eq!(j.finish(), r#"{"a":1,"b":[1,2,{"c":"d"}],"e":{},"f":true}"#);
    }

    #[test]
    fn strings_are_escaped() {
        let mut j = Json::new();
        j.string("a\"b\\c\nd\te\u{1}f");
        assert_eq!(j.finish(), r#""a\"b\\c\nd\te\u0001f""#);
    }

    #[test]
    fn floats_round_and_non_finite_become_null() {
        let mut j = Json::new();
        j.begin_array();
        j.float(1.25, 1).float(2.0, 2).float(f64::NAN, 1);
        j.float(f64::INFINITY, 1).float(-0.04, 1);
        j.end_array();
        assert_eq!(j.finish(), "[1.2,2.00,null,null,-0.0]");
    }

    #[test]
    fn optional_fields_are_omitted_when_unknown() {
        let mut j = Json::new();
        j.begin_object();
        j.field_opt_float("a", None, 1);
        j.field_opt_uint("b", Some(3));
        j.end_object();
        assert_eq!(j.finish(), r#"{"b":3}"#);
    }
}
