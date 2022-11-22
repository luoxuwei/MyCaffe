// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: snapshot.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_snapshot_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_snapshot_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3021000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3021009 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_snapshot_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_snapshot_2eproto {
  static const uint32_t offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_snapshot_2eproto;
namespace MyCaffe {
class Snapshot;
struct SnapshotDefaultTypeInternal;
extern SnapshotDefaultTypeInternal _Snapshot_default_instance_;
class Snapshot_ParamBlok;
struct Snapshot_ParamBlokDefaultTypeInternal;
extern Snapshot_ParamBlokDefaultTypeInternal _Snapshot_ParamBlok_default_instance_;
class Snapshot_ParamBlok_ParamValue;
struct Snapshot_ParamBlok_ParamValueDefaultTypeInternal;
extern Snapshot_ParamBlok_ParamValueDefaultTypeInternal _Snapshot_ParamBlok_ParamValue_default_instance_;
}  // namespace MyCaffe
PROTOBUF_NAMESPACE_OPEN
template<> ::MyCaffe::Snapshot* Arena::CreateMaybeMessage<::MyCaffe::Snapshot>(Arena*);
template<> ::MyCaffe::Snapshot_ParamBlok* Arena::CreateMaybeMessage<::MyCaffe::Snapshot_ParamBlok>(Arena*);
template<> ::MyCaffe::Snapshot_ParamBlok_ParamValue* Arena::CreateMaybeMessage<::MyCaffe::Snapshot_ParamBlok_ParamValue>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace MyCaffe {

// ===================================================================

class Snapshot_ParamBlok_ParamValue final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:MyCaffe.Snapshot.ParamBlok.ParamValue) */ {
 public:
  inline Snapshot_ParamBlok_ParamValue() : Snapshot_ParamBlok_ParamValue(nullptr) {}
  ~Snapshot_ParamBlok_ParamValue() override;
  explicit PROTOBUF_CONSTEXPR Snapshot_ParamBlok_ParamValue(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  Snapshot_ParamBlok_ParamValue(const Snapshot_ParamBlok_ParamValue& from);
  Snapshot_ParamBlok_ParamValue(Snapshot_ParamBlok_ParamValue&& from) noexcept
    : Snapshot_ParamBlok_ParamValue() {
    *this = ::std::move(from);
  }

  inline Snapshot_ParamBlok_ParamValue& operator=(const Snapshot_ParamBlok_ParamValue& from) {
    CopyFrom(from);
    return *this;
  }
  inline Snapshot_ParamBlok_ParamValue& operator=(Snapshot_ParamBlok_ParamValue&& from) noexcept {
    if (this == &from) return *this;
    if (GetOwningArena() == from.GetOwningArena()
  #ifdef PROTOBUF_FORCE_COPY_IN_MOVE
        && GetOwningArena() != nullptr
  #endif  // !PROTOBUF_FORCE_COPY_IN_MOVE
    ) {
      InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return default_instance().GetMetadata().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return default_instance().GetMetadata().reflection;
  }
  static const Snapshot_ParamBlok_ParamValue& default_instance() {
    return *internal_default_instance();
  }
  static inline const Snapshot_ParamBlok_ParamValue* internal_default_instance() {
    return reinterpret_cast<const Snapshot_ParamBlok_ParamValue*>(
               &_Snapshot_ParamBlok_ParamValue_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(Snapshot_ParamBlok_ParamValue& a, Snapshot_ParamBlok_ParamValue& b) {
    a.Swap(&b);
  }
  inline void Swap(Snapshot_ParamBlok_ParamValue* other) {
    if (other == this) return;
  #ifdef PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() != nullptr &&
        GetOwningArena() == other->GetOwningArena()) {
   #else  // PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() == other->GetOwningArena()) {
  #endif  // !PROTOBUF_FORCE_COPY_IN_SWAP
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(Snapshot_ParamBlok_ParamValue* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  Snapshot_ParamBlok_ParamValue* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<Snapshot_ParamBlok_ParamValue>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const Snapshot_ParamBlok_ParamValue& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom( const Snapshot_ParamBlok_ParamValue& from) {
    Snapshot_ParamBlok_ParamValue::MergeImpl(*this, from);
  }
  private:
  static void MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg);
  public:
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  uint8_t* _InternalSerialize(
      uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _impl_._cached_size_.Get(); }

  private:
  void SharedCtor(::PROTOBUF_NAMESPACE_ID::Arena* arena, bool is_message_owned);
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(Snapshot_ParamBlok_ParamValue* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "MyCaffe.Snapshot.ParamBlok.ParamValue";
  }
  protected:
  explicit Snapshot_ParamBlok_ParamValue(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kValueFieldNumber = 1,
  };
  // double value = 1;
  void clear_value();
  double value() const;
  void set_value(double value);
  private:
  double _internal_value() const;
  void _internal_set_value(double value);
  public:

  // @@protoc_insertion_point(class_scope:MyCaffe.Snapshot.ParamBlok.ParamValue)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  struct Impl_ {
    double value_;
    mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  };
  union { Impl_ _impl_; };
  friend struct ::TableStruct_snapshot_2eproto;
};
// -------------------------------------------------------------------

class Snapshot_ParamBlok final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:MyCaffe.Snapshot.ParamBlok) */ {
 public:
  inline Snapshot_ParamBlok() : Snapshot_ParamBlok(nullptr) {}
  ~Snapshot_ParamBlok() override;
  explicit PROTOBUF_CONSTEXPR Snapshot_ParamBlok(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  Snapshot_ParamBlok(const Snapshot_ParamBlok& from);
  Snapshot_ParamBlok(Snapshot_ParamBlok&& from) noexcept
    : Snapshot_ParamBlok() {
    *this = ::std::move(from);
  }

  inline Snapshot_ParamBlok& operator=(const Snapshot_ParamBlok& from) {
    CopyFrom(from);
    return *this;
  }
  inline Snapshot_ParamBlok& operator=(Snapshot_ParamBlok&& from) noexcept {
    if (this == &from) return *this;
    if (GetOwningArena() == from.GetOwningArena()
  #ifdef PROTOBUF_FORCE_COPY_IN_MOVE
        && GetOwningArena() != nullptr
  #endif  // !PROTOBUF_FORCE_COPY_IN_MOVE
    ) {
      InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return default_instance().GetMetadata().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return default_instance().GetMetadata().reflection;
  }
  static const Snapshot_ParamBlok& default_instance() {
    return *internal_default_instance();
  }
  static inline const Snapshot_ParamBlok* internal_default_instance() {
    return reinterpret_cast<const Snapshot_ParamBlok*>(
               &_Snapshot_ParamBlok_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(Snapshot_ParamBlok& a, Snapshot_ParamBlok& b) {
    a.Swap(&b);
  }
  inline void Swap(Snapshot_ParamBlok* other) {
    if (other == this) return;
  #ifdef PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() != nullptr &&
        GetOwningArena() == other->GetOwningArena()) {
   #else  // PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() == other->GetOwningArena()) {
  #endif  // !PROTOBUF_FORCE_COPY_IN_SWAP
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(Snapshot_ParamBlok* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  Snapshot_ParamBlok* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<Snapshot_ParamBlok>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const Snapshot_ParamBlok& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom( const Snapshot_ParamBlok& from) {
    Snapshot_ParamBlok::MergeImpl(*this, from);
  }
  private:
  static void MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg);
  public:
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  uint8_t* _InternalSerialize(
      uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _impl_._cached_size_.Get(); }

  private:
  void SharedCtor(::PROTOBUF_NAMESPACE_ID::Arena* arena, bool is_message_owned);
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(Snapshot_ParamBlok* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "MyCaffe.Snapshot.ParamBlok";
  }
  protected:
  explicit Snapshot_ParamBlok(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  typedef Snapshot_ParamBlok_ParamValue ParamValue;

  // accessors -------------------------------------------------------

  enum : int {
    kParamValueFieldNumber = 7,
    kParamTypeFieldNumber = 1,
    kLayerNameFieldNumber = 2,
    kKernelNFieldNumber = 3,
    kKernelCFieldNumber = 4,
    kKernelHFieldNumber = 5,
    kKernelWFieldNumber = 6,
  };
  // repeated .MyCaffe.Snapshot.ParamBlok.ParamValue param_value = 7;
  int param_value_size() const;
  private:
  int _internal_param_value_size() const;
  public:
  void clear_param_value();
  ::MyCaffe::Snapshot_ParamBlok_ParamValue* mutable_param_value(int index);
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::MyCaffe::Snapshot_ParamBlok_ParamValue >*
      mutable_param_value();
  private:
  const ::MyCaffe::Snapshot_ParamBlok_ParamValue& _internal_param_value(int index) const;
  ::MyCaffe::Snapshot_ParamBlok_ParamValue* _internal_add_param_value();
  public:
  const ::MyCaffe::Snapshot_ParamBlok_ParamValue& param_value(int index) const;
  ::MyCaffe::Snapshot_ParamBlok_ParamValue* add_param_value();
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::MyCaffe::Snapshot_ParamBlok_ParamValue >&
      param_value() const;

  // string param_type = 1;
  void clear_param_type();
  const std::string& param_type() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_param_type(ArgT0&& arg0, ArgT... args);
  std::string* mutable_param_type();
  PROTOBUF_NODISCARD std::string* release_param_type();
  void set_allocated_param_type(std::string* param_type);
  private:
  const std::string& _internal_param_type() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_param_type(const std::string& value);
  std::string* _internal_mutable_param_type();
  public:

  // string layer_name = 2;
  void clear_layer_name();
  const std::string& layer_name() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_layer_name(ArgT0&& arg0, ArgT... args);
  std::string* mutable_layer_name();
  PROTOBUF_NODISCARD std::string* release_layer_name();
  void set_allocated_layer_name(std::string* layer_name);
  private:
  const std::string& _internal_layer_name() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_layer_name(const std::string& value);
  std::string* _internal_mutable_layer_name();
  public:

  // int32 kernel_n = 3;
  void clear_kernel_n();
  int32_t kernel_n() const;
  void set_kernel_n(int32_t value);
  private:
  int32_t _internal_kernel_n() const;
  void _internal_set_kernel_n(int32_t value);
  public:

  // int32 kernel_c = 4;
  void clear_kernel_c();
  int32_t kernel_c() const;
  void set_kernel_c(int32_t value);
  private:
  int32_t _internal_kernel_c() const;
  void _internal_set_kernel_c(int32_t value);
  public:

  // int32 kernel_h = 5;
  void clear_kernel_h();
  int32_t kernel_h() const;
  void set_kernel_h(int32_t value);
  private:
  int32_t _internal_kernel_h() const;
  void _internal_set_kernel_h(int32_t value);
  public:

  // int32 kernel_w = 6;
  void clear_kernel_w();
  int32_t kernel_w() const;
  void set_kernel_w(int32_t value);
  private:
  int32_t _internal_kernel_w() const;
  void _internal_set_kernel_w(int32_t value);
  public:

  // @@protoc_insertion_point(class_scope:MyCaffe.Snapshot.ParamBlok)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  struct Impl_ {
    ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::MyCaffe::Snapshot_ParamBlok_ParamValue > param_value_;
    ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr param_type_;
    ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr layer_name_;
    int32_t kernel_n_;
    int32_t kernel_c_;
    int32_t kernel_h_;
    int32_t kernel_w_;
    mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  };
  union { Impl_ _impl_; };
  friend struct ::TableStruct_snapshot_2eproto;
};
// -------------------------------------------------------------------

class Snapshot final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:MyCaffe.Snapshot) */ {
 public:
  inline Snapshot() : Snapshot(nullptr) {}
  ~Snapshot() override;
  explicit PROTOBUF_CONSTEXPR Snapshot(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  Snapshot(const Snapshot& from);
  Snapshot(Snapshot&& from) noexcept
    : Snapshot() {
    *this = ::std::move(from);
  }

  inline Snapshot& operator=(const Snapshot& from) {
    CopyFrom(from);
    return *this;
  }
  inline Snapshot& operator=(Snapshot&& from) noexcept {
    if (this == &from) return *this;
    if (GetOwningArena() == from.GetOwningArena()
  #ifdef PROTOBUF_FORCE_COPY_IN_MOVE
        && GetOwningArena() != nullptr
  #endif  // !PROTOBUF_FORCE_COPY_IN_MOVE
    ) {
      InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return default_instance().GetMetadata().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return default_instance().GetMetadata().reflection;
  }
  static const Snapshot& default_instance() {
    return *internal_default_instance();
  }
  static inline const Snapshot* internal_default_instance() {
    return reinterpret_cast<const Snapshot*>(
               &_Snapshot_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    2;

  friend void swap(Snapshot& a, Snapshot& b) {
    a.Swap(&b);
  }
  inline void Swap(Snapshot* other) {
    if (other == this) return;
  #ifdef PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() != nullptr &&
        GetOwningArena() == other->GetOwningArena()) {
   #else  // PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() == other->GetOwningArena()) {
  #endif  // !PROTOBUF_FORCE_COPY_IN_SWAP
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(Snapshot* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  Snapshot* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<Snapshot>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const Snapshot& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom( const Snapshot& from) {
    Snapshot::MergeImpl(*this, from);
  }
  private:
  static void MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg);
  public:
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  uint8_t* _InternalSerialize(
      uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _impl_._cached_size_.Get(); }

  private:
  void SharedCtor(::PROTOBUF_NAMESPACE_ID::Arena* arena, bool is_message_owned);
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(Snapshot* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "MyCaffe.Snapshot";
  }
  protected:
  explicit Snapshot(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  typedef Snapshot_ParamBlok ParamBlok;

  // accessors -------------------------------------------------------

  enum : int {
    kParamBlokFieldNumber = 1,
  };
  // repeated .MyCaffe.Snapshot.ParamBlok param_blok = 1;
  int param_blok_size() const;
  private:
  int _internal_param_blok_size() const;
  public:
  void clear_param_blok();
  ::MyCaffe::Snapshot_ParamBlok* mutable_param_blok(int index);
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::MyCaffe::Snapshot_ParamBlok >*
      mutable_param_blok();
  private:
  const ::MyCaffe::Snapshot_ParamBlok& _internal_param_blok(int index) const;
  ::MyCaffe::Snapshot_ParamBlok* _internal_add_param_blok();
  public:
  const ::MyCaffe::Snapshot_ParamBlok& param_blok(int index) const;
  ::MyCaffe::Snapshot_ParamBlok* add_param_blok();
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::MyCaffe::Snapshot_ParamBlok >&
      param_blok() const;

  // @@protoc_insertion_point(class_scope:MyCaffe.Snapshot)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  struct Impl_ {
    ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::MyCaffe::Snapshot_ParamBlok > param_blok_;
    mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  };
  union { Impl_ _impl_; };
  friend struct ::TableStruct_snapshot_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// Snapshot_ParamBlok_ParamValue

// double value = 1;
inline void Snapshot_ParamBlok_ParamValue::clear_value() {
  _impl_.value_ = 0;
}
inline double Snapshot_ParamBlok_ParamValue::_internal_value() const {
  return _impl_.value_;
}
inline double Snapshot_ParamBlok_ParamValue::value() const {
  // @@protoc_insertion_point(field_get:MyCaffe.Snapshot.ParamBlok.ParamValue.value)
  return _internal_value();
}
inline void Snapshot_ParamBlok_ParamValue::_internal_set_value(double value) {
  
  _impl_.value_ = value;
}
inline void Snapshot_ParamBlok_ParamValue::set_value(double value) {
  _internal_set_value(value);
  // @@protoc_insertion_point(field_set:MyCaffe.Snapshot.ParamBlok.ParamValue.value)
}

// -------------------------------------------------------------------

// Snapshot_ParamBlok

// string param_type = 1;
inline void Snapshot_ParamBlok::clear_param_type() {
  _impl_.param_type_.ClearToEmpty();
}
inline const std::string& Snapshot_ParamBlok::param_type() const {
  // @@protoc_insertion_point(field_get:MyCaffe.Snapshot.ParamBlok.param_type)
  return _internal_param_type();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void Snapshot_ParamBlok::set_param_type(ArgT0&& arg0, ArgT... args) {
 
 _impl_.param_type_.Set(static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:MyCaffe.Snapshot.ParamBlok.param_type)
}
inline std::string* Snapshot_ParamBlok::mutable_param_type() {
  std::string* _s = _internal_mutable_param_type();
  // @@protoc_insertion_point(field_mutable:MyCaffe.Snapshot.ParamBlok.param_type)
  return _s;
}
inline const std::string& Snapshot_ParamBlok::_internal_param_type() const {
  return _impl_.param_type_.Get();
}
inline void Snapshot_ParamBlok::_internal_set_param_type(const std::string& value) {
  
  _impl_.param_type_.Set(value, GetArenaForAllocation());
}
inline std::string* Snapshot_ParamBlok::_internal_mutable_param_type() {
  
  return _impl_.param_type_.Mutable(GetArenaForAllocation());
}
inline std::string* Snapshot_ParamBlok::release_param_type() {
  // @@protoc_insertion_point(field_release:MyCaffe.Snapshot.ParamBlok.param_type)
  return _impl_.param_type_.Release();
}
inline void Snapshot_ParamBlok::set_allocated_param_type(std::string* param_type) {
  if (param_type != nullptr) {
    
  } else {
    
  }
  _impl_.param_type_.SetAllocated(param_type, GetArenaForAllocation());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (_impl_.param_type_.IsDefault()) {
    _impl_.param_type_.Set("", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  // @@protoc_insertion_point(field_set_allocated:MyCaffe.Snapshot.ParamBlok.param_type)
}

// string layer_name = 2;
inline void Snapshot_ParamBlok::clear_layer_name() {
  _impl_.layer_name_.ClearToEmpty();
}
inline const std::string& Snapshot_ParamBlok::layer_name() const {
  // @@protoc_insertion_point(field_get:MyCaffe.Snapshot.ParamBlok.layer_name)
  return _internal_layer_name();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void Snapshot_ParamBlok::set_layer_name(ArgT0&& arg0, ArgT... args) {
 
 _impl_.layer_name_.Set(static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:MyCaffe.Snapshot.ParamBlok.layer_name)
}
inline std::string* Snapshot_ParamBlok::mutable_layer_name() {
  std::string* _s = _internal_mutable_layer_name();
  // @@protoc_insertion_point(field_mutable:MyCaffe.Snapshot.ParamBlok.layer_name)
  return _s;
}
inline const std::string& Snapshot_ParamBlok::_internal_layer_name() const {
  return _impl_.layer_name_.Get();
}
inline void Snapshot_ParamBlok::_internal_set_layer_name(const std::string& value) {
  
  _impl_.layer_name_.Set(value, GetArenaForAllocation());
}
inline std::string* Snapshot_ParamBlok::_internal_mutable_layer_name() {
  
  return _impl_.layer_name_.Mutable(GetArenaForAllocation());
}
inline std::string* Snapshot_ParamBlok::release_layer_name() {
  // @@protoc_insertion_point(field_release:MyCaffe.Snapshot.ParamBlok.layer_name)
  return _impl_.layer_name_.Release();
}
inline void Snapshot_ParamBlok::set_allocated_layer_name(std::string* layer_name) {
  if (layer_name != nullptr) {
    
  } else {
    
  }
  _impl_.layer_name_.SetAllocated(layer_name, GetArenaForAllocation());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (_impl_.layer_name_.IsDefault()) {
    _impl_.layer_name_.Set("", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  // @@protoc_insertion_point(field_set_allocated:MyCaffe.Snapshot.ParamBlok.layer_name)
}

// int32 kernel_n = 3;
inline void Snapshot_ParamBlok::clear_kernel_n() {
  _impl_.kernel_n_ = 0;
}
inline int32_t Snapshot_ParamBlok::_internal_kernel_n() const {
  return _impl_.kernel_n_;
}
inline int32_t Snapshot_ParamBlok::kernel_n() const {
  // @@protoc_insertion_point(field_get:MyCaffe.Snapshot.ParamBlok.kernel_n)
  return _internal_kernel_n();
}
inline void Snapshot_ParamBlok::_internal_set_kernel_n(int32_t value) {
  
  _impl_.kernel_n_ = value;
}
inline void Snapshot_ParamBlok::set_kernel_n(int32_t value) {
  _internal_set_kernel_n(value);
  // @@protoc_insertion_point(field_set:MyCaffe.Snapshot.ParamBlok.kernel_n)
}

// int32 kernel_c = 4;
inline void Snapshot_ParamBlok::clear_kernel_c() {
  _impl_.kernel_c_ = 0;
}
inline int32_t Snapshot_ParamBlok::_internal_kernel_c() const {
  return _impl_.kernel_c_;
}
inline int32_t Snapshot_ParamBlok::kernel_c() const {
  // @@protoc_insertion_point(field_get:MyCaffe.Snapshot.ParamBlok.kernel_c)
  return _internal_kernel_c();
}
inline void Snapshot_ParamBlok::_internal_set_kernel_c(int32_t value) {
  
  _impl_.kernel_c_ = value;
}
inline void Snapshot_ParamBlok::set_kernel_c(int32_t value) {
  _internal_set_kernel_c(value);
  // @@protoc_insertion_point(field_set:MyCaffe.Snapshot.ParamBlok.kernel_c)
}

// int32 kernel_h = 5;
inline void Snapshot_ParamBlok::clear_kernel_h() {
  _impl_.kernel_h_ = 0;
}
inline int32_t Snapshot_ParamBlok::_internal_kernel_h() const {
  return _impl_.kernel_h_;
}
inline int32_t Snapshot_ParamBlok::kernel_h() const {
  // @@protoc_insertion_point(field_get:MyCaffe.Snapshot.ParamBlok.kernel_h)
  return _internal_kernel_h();
}
inline void Snapshot_ParamBlok::_internal_set_kernel_h(int32_t value) {
  
  _impl_.kernel_h_ = value;
}
inline void Snapshot_ParamBlok::set_kernel_h(int32_t value) {
  _internal_set_kernel_h(value);
  // @@protoc_insertion_point(field_set:MyCaffe.Snapshot.ParamBlok.kernel_h)
}

// int32 kernel_w = 6;
inline void Snapshot_ParamBlok::clear_kernel_w() {
  _impl_.kernel_w_ = 0;
}
inline int32_t Snapshot_ParamBlok::_internal_kernel_w() const {
  return _impl_.kernel_w_;
}
inline int32_t Snapshot_ParamBlok::kernel_w() const {
  // @@protoc_insertion_point(field_get:MyCaffe.Snapshot.ParamBlok.kernel_w)
  return _internal_kernel_w();
}
inline void Snapshot_ParamBlok::_internal_set_kernel_w(int32_t value) {
  
  _impl_.kernel_w_ = value;
}
inline void Snapshot_ParamBlok::set_kernel_w(int32_t value) {
  _internal_set_kernel_w(value);
  // @@protoc_insertion_point(field_set:MyCaffe.Snapshot.ParamBlok.kernel_w)
}

// repeated .MyCaffe.Snapshot.ParamBlok.ParamValue param_value = 7;
inline int Snapshot_ParamBlok::_internal_param_value_size() const {
  return _impl_.param_value_.size();
}
inline int Snapshot_ParamBlok::param_value_size() const {
  return _internal_param_value_size();
}
inline void Snapshot_ParamBlok::clear_param_value() {
  _impl_.param_value_.Clear();
}
inline ::MyCaffe::Snapshot_ParamBlok_ParamValue* Snapshot_ParamBlok::mutable_param_value(int index) {
  // @@protoc_insertion_point(field_mutable:MyCaffe.Snapshot.ParamBlok.param_value)
  return _impl_.param_value_.Mutable(index);
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::MyCaffe::Snapshot_ParamBlok_ParamValue >*
Snapshot_ParamBlok::mutable_param_value() {
  // @@protoc_insertion_point(field_mutable_list:MyCaffe.Snapshot.ParamBlok.param_value)
  return &_impl_.param_value_;
}
inline const ::MyCaffe::Snapshot_ParamBlok_ParamValue& Snapshot_ParamBlok::_internal_param_value(int index) const {
  return _impl_.param_value_.Get(index);
}
inline const ::MyCaffe::Snapshot_ParamBlok_ParamValue& Snapshot_ParamBlok::param_value(int index) const {
  // @@protoc_insertion_point(field_get:MyCaffe.Snapshot.ParamBlok.param_value)
  return _internal_param_value(index);
}
inline ::MyCaffe::Snapshot_ParamBlok_ParamValue* Snapshot_ParamBlok::_internal_add_param_value() {
  return _impl_.param_value_.Add();
}
inline ::MyCaffe::Snapshot_ParamBlok_ParamValue* Snapshot_ParamBlok::add_param_value() {
  ::MyCaffe::Snapshot_ParamBlok_ParamValue* _add = _internal_add_param_value();
  // @@protoc_insertion_point(field_add:MyCaffe.Snapshot.ParamBlok.param_value)
  return _add;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::MyCaffe::Snapshot_ParamBlok_ParamValue >&
Snapshot_ParamBlok::param_value() const {
  // @@protoc_insertion_point(field_list:MyCaffe.Snapshot.ParamBlok.param_value)
  return _impl_.param_value_;
}

// -------------------------------------------------------------------

// Snapshot

// repeated .MyCaffe.Snapshot.ParamBlok param_blok = 1;
inline int Snapshot::_internal_param_blok_size() const {
  return _impl_.param_blok_.size();
}
inline int Snapshot::param_blok_size() const {
  return _internal_param_blok_size();
}
inline void Snapshot::clear_param_blok() {
  _impl_.param_blok_.Clear();
}
inline ::MyCaffe::Snapshot_ParamBlok* Snapshot::mutable_param_blok(int index) {
  // @@protoc_insertion_point(field_mutable:MyCaffe.Snapshot.param_blok)
  return _impl_.param_blok_.Mutable(index);
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::MyCaffe::Snapshot_ParamBlok >*
Snapshot::mutable_param_blok() {
  // @@protoc_insertion_point(field_mutable_list:MyCaffe.Snapshot.param_blok)
  return &_impl_.param_blok_;
}
inline const ::MyCaffe::Snapshot_ParamBlok& Snapshot::_internal_param_blok(int index) const {
  return _impl_.param_blok_.Get(index);
}
inline const ::MyCaffe::Snapshot_ParamBlok& Snapshot::param_blok(int index) const {
  // @@protoc_insertion_point(field_get:MyCaffe.Snapshot.param_blok)
  return _internal_param_blok(index);
}
inline ::MyCaffe::Snapshot_ParamBlok* Snapshot::_internal_add_param_blok() {
  return _impl_.param_blok_.Add();
}
inline ::MyCaffe::Snapshot_ParamBlok* Snapshot::add_param_blok() {
  ::MyCaffe::Snapshot_ParamBlok* _add = _internal_add_param_blok();
  // @@protoc_insertion_point(field_add:MyCaffe.Snapshot.param_blok)
  return _add;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::MyCaffe::Snapshot_ParamBlok >&
Snapshot::param_blok() const {
  // @@protoc_insertion_point(field_list:MyCaffe.Snapshot.param_blok)
  return _impl_.param_blok_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------

// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace MyCaffe

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_snapshot_2eproto
